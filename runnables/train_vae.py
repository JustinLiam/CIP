import pytorch_lightning as pl
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger
import importlib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hydra.utils import get_original_cwd
import pickle

from src.utils.utils import set_seed
from src.utils.utils import get_checkpoint_filename, evaluate, get_absolute_path, log_data_seed, clear_tfevents, add_float_treatment, repeat_static, to_float, count_parameters, to_double

import warnings
from src.utils.helper_functions import write_csv, check_csv
warnings.filterwarnings("ignore")

# 告诉 Hydra（配置/运行管理库）在报错时输出**完整堆栈信息**（full traceback），而不是简化后的错误信息。
os.environ['HYDRA_FULL_ERROR'] = '1'
# 日志系统的基础配置
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# 使用 Hydra 的主函数装饰器，指定配置文件名和路径
@hydra.main(config_name='config.yaml', config_path='../configs/')
# DictConfig 是 OmegaConf 的类型，表示“像字典一样访问的配置树”，所以你能写 args.exp.seed、args["model"]["_target_"] 这种访问方式。
# args 不一定只来自单个 config.yaml：Hydra 可能还会根据 defaults 组合多个配置组，并应用命令行覆盖（override），最后得到一个合并后的 DictConfig 传进来。
# 你代码里 logger.info('\\n' + OmegaConf.to_yaml(args, resolve=True)) 打印出的内容，就是最终传入 main 的这份配置（已解析后的形式）。
def main(args: DictConfig):
    # Basic setup
    OmegaConf.set_struct(args, False)  # 关闭 OmegaConf/Hydra 配置对象 args 的“结构体模式（struct mode），允许动态添加新字段
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))  #这行会在日志里打印类似“本次训练用的完整配置清单”
    set_seed(args.exp.seed)

    # Check optimization requirements
    current_dir = get_absolute_path('./')
    csv_dir = current_dir + f'/{args.exp.test}'
    #
    optimize_interventions = True
    if not check_csv(csv_dir, args):
        optimize_interventions = False

    # Data loading and processing
    original_cwd = get_original_cwd()
    args['exp']['processed_data_dir'] = os.path.join(original_cwd, args['exp']['processed_data_dir'])
    
    path = os.path.join(args['exp']['processed_data_dir'], f"seed_{args.exp.seed}.pkl")  # 这一行是在构造一个缓存文件的完整路径，用于把“当前随机种子 (args.exp.seed) 对应的数据预处理结果”读写到磁盘上，避免每次运行都重新处理数据。
    if True:
        os.makedirs(args['exp']['processed_data_dir'], exist_ok=True)
        dataset_collection = instantiate(args.dataset, _recursive_=True)
        dataset_collection.process_data_multi()
        dataset_collection = to_float(dataset_collection)
        if args['dataset']['static_size'] > 0:
            dims = len(dataset_collection.train_f.data['static_features'].shape)
            if dims == 2:
                dataset_collection = repeat_static(dataset_collection)  # 把静态特征从二维扩展到三维，方便后续模型处理
    else:
        with open(path, 'rb') as file:
            dataset_collection = pickle.load(file)
            if args['dataset']['static_size'] > 0:
                dims = len(dataset_collection.train_f.data['static_features'].shape)
                if dims == 2:
                    dataset_collection = repeat_static(dataset_collection)

    # Model initialization and training
    module_path, class_name = args["model"]["_target_"].rsplit('.', 1)  # module_path == src.models.vae_model, class_name == VAEModel
    model_class = getattr(importlib.import_module(module_path), class_name)  # 导入 src.models.vae_model 这个模块，再用 getattr(..., class_name) 从模块里取出 VAEModel 这个类
    model = model_class(args, dataset_collection)

    # Setup model directory with seed subdirectory
    current_dir = get_absolute_path('./')
    model_dir = os.path.join(current_dir, 'models', f'{args.exp.seed}')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'model.ckpt')
    
    if os.path.exists(model_path):
        # Load existing model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded existing model from {model_path}")
    else:
        # Train new model
        logger_board = TensorBoardLogger(save_dir=model_dir, name='', version='')
        
        trainer = pl.Trainer(
            logger=logger_board,
            max_epochs=args.exp.epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            devices=args.exp.gpus,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=model_dir,
                    monitor='val_loss',
                    filename='model',
                    save_top_k=1,
                    mode='min'
                )
            ]
        )
        
        trainer.fit(model)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Trained and saved new model to {model_path}")

    # Optimization phase
    if args.exp.rank:
        if args.dataset.name == 'mimic3_real' or (args.dataset.name == 'tumor_generator' and args.dataset.coeff in [1,2,3,4]):
            model.optimize_interventions_discrete()
    else:
        if not optimize_interventions:
            return
            
        num_iterations = 100
        result = model.optimize_interventions(num_iterations=num_iterations, learning_rate=0.01, batch_size=128)
        
        # Save results
        result_path = os.path.join(model_dir, "results.txt")
        with open(result_path, 'a') as f:
            f.write(result)
        
        write_csv(result, csv_dir, args)
        return result

if __name__ == "__main__":
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    
    warnings.filterwarnings("ignore")
    os.environ['HYDRA_FULL_ERROR'] = '1'
    OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)
    main()