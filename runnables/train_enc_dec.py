import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
# from pytorch_lightning.utilities.seed import seed_everything

from pytorch_lightning.callbacks import LearningRateMonitor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.utils import AlphaRise, FilteringMlFlowLogger
from src.utils.utils import set_seed as seed_everything

from src.utils.utils import get_checkpoint_filename, evaluate, get_absolute_path, log_data_seed, clear_tfevents, add_float_treatment, repeat_static, to_float, count_parameters, set_seed, to_double
import pandas as pd
import time
import pickle
from hydra.utils import get_original_cwd

from src.utils.helper_functions import write_csv, check_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# torch.set_default_dtype(torch.double)
torch.set_default_dtype(torch.float32)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)
@hydra.main(config_name=f'config.yaml', config_path='../configs/')
def main(args: DictConfig):
    """
    Training / evaluation script for models with encoder-decoder structure: CRN, EDCT (Encoder-Decoder Causal Transformer)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data
    seed_everything(args.exp.seed)
    current_dir = get_absolute_path('./')
    csv_dir = current_dir + f'/{args.exp.test}'
    optimize_interventions = True
    if not check_csv(csv_dir, args):
        optimize_interventions = False

    if 'data_seed' in args.dataset:
        # always use the same data seed
        original_cwd = get_original_cwd()
        args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
        # path = os.path.join(args['exp']['processed_data_dir'], f"seed_10.pkl")
        # seed = args.dataset.data_seed
        seed = 10
        path = os.path.join(args['exp']['processed_data_dir'], f"seed_{seed}.pkl")
        if not os.path.exists(path):
            print(f'{path} does not exist. Creating it now.')
            os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
            # assert args.exp.seed == 10
            assert args.exp.seed == seed
            dataset_collection = instantiate(args.dataset, _recursive_=True)
            dataset_collection.process_data_encoder()
            # with open(path, 'wb') as file:
            #     pickle.dump(dataset_collection, file)
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
    encoder_callbacks, decoder_callbacks = [AlphaRise(rate=args.exp.alpha_rate)], [AlphaRise(rate=args.exp.alpha_rate)]

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=['encoder', 'decoder'],
                                           experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== Initialisation & Training of encoder ==============================
    encoder = instantiate(args.model.encoder, args, dataset_collection, _recursive_=False)
    if args.model.encoder.tune_hparams:
        encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

    encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                              callbacks=encoder_callbacks, terminate_on_nan=True, precision=32)
    start_time = time.time()

    
    model_dir = os.path.join(current_dir, 'models' + f'/{args.exp.seed}')
    # check if encoder model exists
    checkpoint_encoder = model_dir + '/' + 'encoder.ckpt'
    if os.path.exists(checkpoint_encoder):
        # encoder = encoder.load_from_checkpoint(checkpoint_encoder)
        checkpoint = torch.load(checkpoint_encoder)
        encoder.load_state_dict(checkpoint)
    else:
        encoder_trainer.fit(encoder)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(encoder.state_dict(), checkpoint_encoder)

    training_time = time.time() - start_time
    encoder_results = {}


    # ============================== Initialisation & Training of decoder ==============================
    if args.model.train_decoder:
        if args.exp.logging:
            mlf_logger.filter_submodel = 'encoder'  # Disable Logging to Mlflow
        decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)

        if args.model.decoder.tune_hparams:
            decoder.finetune(resources_per_trial=args.model.decoder.resources_per_trial)

        decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                  callbacks=decoder_callbacks, terminate_on_nan=True,
                                  precision=32,)
        start_time = time.time()

        # check if decoder model exists
        checkpoint_decoder = model_dir + '/' + 'decoder.ckpt'
        if os.path.exists(checkpoint_decoder):
            checkpoint = torch.load(checkpoint_decoder)
            decoder.load_state_dict(checkpoint)
        else:
            decoder_trainer.fit(decoder)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(decoder.state_dict(), checkpoint_decoder)
    
    num_iterations = 100 
    try:
        if args.dataset.coeff == 8:
            num_iterations = 150
    except:
        pass 
    batch_size = 128
    learning_rate = 0.01

    if args.exp.rank:
        if args.dataset.name == 'mimic3_real' or (args.dataset.name == 'tumor_generator' and args.dataset.coeff in [1,2,3,4]):
            # if args.exp.seed == 10:
            if True:
                decoder.optimize_interventions_discrete(encoder=encoder)
    # return
    else:
        if not optimize_interventions:
            return 
        results = decoder.optimize_interventions(encoder=encoder, num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)

        result_path = model_dir + '/' + "results.txt"
        with open(result_path, 'a') as f:
            f.write(results)

        write_csv(results, csv_dir, args)
            
        return results


if __name__ == "__main__":
    main()

