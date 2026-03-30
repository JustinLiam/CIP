import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import time
import argparse
import pandas as pd
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger
import omegaconf
from pytorch_lightning.loggers import MLFlowLogger
import importlib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.utils import set_seed, AlphaRise
from src.utils.utils import get_checkpoint_filename, evaluate, get_absolute_path, log_data_seed, clear_tfevents, add_float_treatment, repeat_static, to_float, count_parameters, to_double
import warnings
from hydra.utils import get_original_cwd
import pickle
from src.utils.helper_functions import write_csv, check_csv

warnings.filterwarnings("ignore", category=UserWarning, module='fvcore.*')

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)
@hydra.main(config_name=f'config.yaml', config_path='../configs/')
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    print('\n' + OmegaConf.to_yaml(args, resolve=True))
    set_seed(args['exp']['seed'])
    # set checkpoint
    current_date = time.strftime("%Y-%m-%d", time.localtime())
    model_name = args['model']['name']
    current_dir = get_absolute_path('./') 
    print(current_dir)
    csv_dir = current_dir + f'/{args.exp.test}'
    optimize_interventions = True
    if not check_csv(csv_dir, args):
        optimize_interventions = False
        # return None
    # exit()
    # get current time as pytorch-lightning version
    version = time.strftime("%H-%M-%S", time.localtime())
    
    original_cwd = get_original_cwd()
    args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
    path = os.path.join(args['exp']['processed_data_dir'], f"seed_{args['exp']['seed']}.pkl")
    if 'data_seed' in args.dataset:
        original_cwd = get_original_cwd()
        args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
        path = os.path.join(args['exp']['processed_data_dir'], f"seed_10.pkl")
        if not os.path.exists(path):
            print(f'{path} does not exist. Creating it now.')
            os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
            assert args.exp.seed == 10
            dataset_collection = instantiate(args.dataset, _recursive_=True)
            dataset_collection.process_data_multi()
            dataset_collection = to_float(dataset_collection)
            # with open(path, 'wb') as file:
            #     pickle.dump(dataset_collection, file)
        else:
            with open(path, 'rb') as file:
                dataset_collection = pickle.load(file)
                
    elif args['exp']['load_data']:
        original_cwd = get_original_cwd()
        args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
        path = os.path.join(args['exp']['processed_data_dir'], f"seed_{args['exp']['seed']}.pkl")
        if not os.path.exists(path):
            print(f'{path} does not exist. Creating it now.')
            os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
            dataset_collection = instantiate(args.dataset, _recursive_=True)
            dataset_collection.process_data_multi()
            dataset_collection = to_float(dataset_collection)
            # with open(path, 'wb') as file:
            #     pickle.dump(dataset_collection, file)
        else:
            with open(path, 'rb') as file:
                dataset_collection = pickle.load(file)
    else:
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        dataset_collection.process_data_multi()
        dataset_collection = to_float(dataset_collection)

    if args['dataset']['static_size'] > 0:
        # check if the dim of static features equals to 2
        dims = len(dataset_collection.train_f.data['static_features'].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    # set early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args['exp']['patience'],
        mode='min')
    # set learning rate monitor
    # lr_monitor = LearningRateMonitor(logging_interval='epoch')
    class_path = args["model"]["_target_"]
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model = cls(dataset_collection, args)
    count_parameters(model, logger=logger) 
    # set Logger for pytorch-lightning
    logger_board = TensorBoardLogger(
        save_dir=current_dir, 
        name='', 
        version='')
    if args['exp']['logging']:
        mlf_logger = MLFlowLogger(
            experiment_name=f'{args["model"]["name"]}/{args["exp"]["mode"]}/{args["exp"]["name"]}',
            tracking_uri="http://localhost:5000",
            # save_dir = get_absolute_path('mlruns'),
        )
    else:
        mlf_logger = None
    if args.exp.logging:
        mlf_logger.log_metrics({'ymean': dataset_collection.train_f.data['outputs'].mean()})
        experiment_id = mlf_logger.experiment_id
        run_id = mlf_logger.run_id
        dirpath = os.path.join('checkpoints/', experiment_id, run_id)
    else:
        dirpath = None
    # set checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_loss:.5f}',
        save_top_k=1,
        mode='min'
    )
    # set trainer
    alpharise = AlphaRise()
    trainer = pl.Trainer(
        logger=logger_board,
        max_epochs=args['exp']['epochs'],
        # enable_progress_bar=False,
        # enable_model_summary=False, 
        accelerator='gpu',
        devices=args['exp']['gpus'],
        callbacks=[early_stop_callback, checkpoint_callback, alpharise],
        precision=32,
    )

    current_dir = get_absolute_path('./')
    model_dir = os.path.join(current_dir, 'models' + f'/{args.exp.seed}')
    checkpoint_path = model_dir + '/' + 'model.ckpt'

    if os.path.exists(checkpoint_path):
        # encoder = encoder.load_from_checkpoint(checkpoint_encoder)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        trainer.fit(model)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
    
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
                model.optimize_interventions_discrete()
        # return
    else:
        if not optimize_interventions:
            return 
        result = model.optimize_interventions(num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)

        result_path = model_dir + '/' + "results.txt"
        with open(result_path, 'a') as f:
            f.write(result)
        
        write_csv(result, csv_dir, args)

        return result
    

if __name__ == "__main__":
    main()

