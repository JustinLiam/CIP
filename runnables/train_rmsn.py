import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

# from src.models.utils import AlphaRise, FilteringMlFlowLogger
# from src.models.rmsn import RMSN

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines.utils import AlphaRise, FilteringMlFlowLogger
from src.baselines.rmsn import RMSN
from src.utils.utils import set_seed as seed_everything
from src.utils.utils import get_checkpoint_filename, evaluate, get_absolute_path, log_data_seed, clear_tfevents, add_float_treatment, repeat_static, to_float, count_parameters, set_seed, to_double
import pandas as pd
import time
import pickle
from hydra.utils import get_original_cwd
from src.utils.helper_functions import write_csv, check_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)
@hydra.main(config_name=f'config.yaml', config_path='../configs/')
def main(args: DictConfig):
    """
    Training / evaluation script for RMSNs
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data to calculate dim_outcomes, dim_treatments, dim_vitals and dim_static_features
    seed_everything(args.exp.seed)
    # dataset_collection = instantiate(args.dataset, _recursive_=True)
    # assert args.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
    # dataset_collection.process_data_encoder()
    current_dir = get_absolute_path('./')
    csv_dir = current_dir + f'/{args.exp.test}'
    optimize_interventions = True
    if not check_csv(csv_dir, args):
        optimize_interventions = False

    if 'data_seed' in args.dataset:
        # always use the same data seed
        original_cwd = get_original_cwd()
        args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
        # seed = args.dataset.data_seed
        seed = 10
        path = os.path.join(args['exp']['processed_data_dir'], f"seed_{seed}.pkl")
        if not os.path.exists(path):
            print(f'{path} does not exist. Creating it now.')
            os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
            assert args.exp.seed == seed
            dataset_collection = instantiate(args.dataset, _recursive_=True)
            dataset_collection.process_data_encoder()
            with open(path, 'wb') as file:
                pickle.dump(dataset_collection, file)
        else:
            with open(path, 'rb') as file:
                dataset_collection = pickle.load(file)
            # dataset_collection = to_double(dataset_collection)
    else:
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        dataset_collection.process_data_encoder()
        dataset_collection = to_float(dataset_collection)

    append_results = {}
    append_results['mean_y'] = dataset_collection.train_f.data['outputs'].mean()

    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    prop_treatment_callbacks, propensity_history_callbacks, encoder_callbacks, decoder_callbacks = [], [], [], []

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=RMSN.possible_model_types, experiment_name=experiment_name,
                                           tracking_uri=args.exp.mlflow_uri)
        encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        prop_treatment_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        propensity_history_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    else:
        mlf_logger = None

    # ============================== Nominator (treatment propensity network) ==============================
    propensity_treatment = instantiate(args.model.propensity_treatment, args, dataset_collection, _recursive_=False)
    if args.model.propensity_treatment.tune_hparams:
        propensity_treatment.finetune(resources_per_trial=args.model.propensity_treatment.resources_per_trial)

    propensity_treatment_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                           logger=mlf_logger,
                                           max_epochs=args.exp.max_epochs,
                                           callbacks=prop_treatment_callbacks,
                                           gradient_clip_val=args.model.propensity_treatment.max_grad_norm,
                                           terminate_on_nan=True,
                                           precision=32,)

    current_dir = get_absolute_path('./')
    model_dir = os.path.join(current_dir, 'models' + f'/{args.exp.seed}')

    propensity_treatment_path = os.path.join(model_dir, 'propensity_treatment.ckpt') 
    if os.path.exists(propensity_treatment_path):  
        checkpoint = torch.load(propensity_treatment_path)
        propensity_treatment.load_state_dict(checkpoint)     
    else:                     
        propensity_treatment_trainer.fit(propensity_treatment)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(propensity_treatment.state_dict(), propensity_treatment_path)

    # ============================== Denominator (history propensity network) ==============================
    propensity_history = instantiate(args.model.propensity_history, args, dataset_collection, _recursive_=False)
    if args.model.propensity_history.tune_hparams:
        propensity_history.finetune(resources_per_trial=args.model.propensity_history.resources_per_trial)

    propensity_history_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                         logger=mlf_logger,
                                         max_epochs=args.exp.max_epochs,
                                         callbacks=propensity_history_callbacks,
                                         gradient_clip_val=args.model.propensity_history.max_grad_norm,
                                         terminate_on_nan=True,
                                         precision=32,)

    propensity_history_path = os.path.join(model_dir, 'propensity_history.ckpt') 
    if os.path.exists(propensity_history_path):  
        checkpoint = torch.load(propensity_history_path)
        propensity_history.load_state_dict(checkpoint)     
    else:                     
        propensity_history_trainer.fit(propensity_history)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(propensity_history.state_dict(), propensity_history_path)

    # ============================== Initialisation & Training of Encoder ==============================
    encoder = instantiate(args.model.encoder, args, propensity_treatment, propensity_history, dataset_collection,
                          _recursive_=False)
    if args.model.encoder.tune_hparams:
        encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

    encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                              logger=mlf_logger,
                              max_epochs=args.exp.max_epochs,
                              callbacks=encoder_callbacks,
                              gradient_clip_val=args.model.encoder.max_grad_norm,
                              terminate_on_nan=True,
                              precision=32,)

    encoder_path = os.path.join(model_dir, 'encoder.ckpt') 
    if os.path.exists(encoder_path):  
        checkpoint = torch.load(encoder_path)
        encoder.load_state_dict(checkpoint)     
    else:                     
        encoder_trainer.fit(encoder)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(encoder.state_dict(), encoder_path)

    # ============================== Initialisation & Training of Decoder ==============================
    if args.model.train_decoder:
        decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)

        if args.model.decoder.tune_hparams:
            decoder.finetune(resources_per_trial=args.model.decoder.resources_per_trial)

        decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                  logger=mlf_logger,
                                  max_epochs=args.exp.max_epochs,
                                  gradient_clip_val=args.model.decoder.max_grad_norm,
                                  callbacks=decoder_callbacks,
                                  terminate_on_nan=True,
                                  precision=32,)

        decoder_path = os.path.join(model_dir, 'decoder.ckpt') 
        if os.path.exists(decoder_path):  
            checkpoint = torch.load(decoder_path)
            decoder.load_state_dict(checkpoint)     
        else:                     
            decoder_trainer.fit(decoder)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(decoder.state_dict(), decoder_path)

    num_iterations = 100
    try:
        if args.dataset.coeff == 8:
            num_iterations = 150
    except:
        pass 
    batch_size = 128
    learning_rate = 0.01

    if args.exp.rank:
        if args.dataset.name == 'mimic3_real' or (args.dataset.name == 'tumor_generator' and args.dataset.coeff  in [1,2,3,4]):
            # if args.exp.seed == 10:
            if True:
                decoder.optimize_interventions_discrete(encoder=encoder)
                
    else:
        if not optimize_interventions:
            return 
        result = decoder.optimize_interventions(encoder=encoder, num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)

        result_path = model_dir + '/' + "results.txt"
        with open(result_path, 'a') as f:
            f.write(result)
        
        write_csv(result, csv_dir, args)

        return result
    
if __name__ == "__main__":
    main()