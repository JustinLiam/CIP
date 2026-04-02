import torch
import torch.nn.functional as F
import yaml
import os
import numpy as np
from typing import Optional, Sequence, Union
import random
import glob
from pytorch_lightning.callbacks import Callback
import logging
logger = logging.getLogger(__name__)

# set random seed
def set_seed(seed: int = 42, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

# unscale the data
def unscale_cancer_data(data, config):
    output_stds = config['dataset']['output_stds']
    output_means = config['dataset']['output_means']
    return data * output_stds + output_means

# get checkpoint filename based on config
def get_checkpoint_filename(config):
    # filename = 'model_{}_lr_{}_bs_{}_hd_{}_fc_{}_nl_{}_dr_{}_wd_{}_sch-patience_{}_patience_{}_factor_{}'.format(config['model'], config['lr'], config['batch_size'], config['hidden_dim_lstm'], config['fc_hidden_units'], config['num_layers'], config['dropout'], config['weight_decay'], config['sch_patience'], config['patience'], config['factor'])
    # with open(get_absolute_path('configs/{}/notation.yaml'.format(config['data'])), 'r') as f:
    #     notation = yaml.safe_load(f)
    # # notation = yaml.safe_load(get_absolute_path('configs/{}/notation.yaml'.format(config['data'])))
    # filename = []
    # for key, value in notation.items():
    #     if value in config:
    #         filename.append('{}_{}'.format(key, config[value]))
    # filename = '_'.join(filename)
    filename = 'model_{}'.format(config['model']['name'])
    return filename

# get absolute path of a dir
def get_absolute_path(path):
    # get current working directory
    cwd = os.getcwd()
    # get absolute path
    absolute_path = os.path.join(cwd, path)
    return absolute_path

# evaluate the model on cancer dataset
def evaluate_cancer(model, data_loader, config):
    rmses, mses = evaluate_cancer_at_each_follow_up_time(model, data_loader, config)
    rmse = (torch.sqrt(torch.mean(mses))) / 1150 * 100
    return rmse.item(), [rmses[i].item() for i in range(len(rmses))]

# evaluate the model at each follow-up time
def evaluate_cancer_at_each_follow_up_time(model, data_loader, config):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_unscaled_outputs = []
        all_active_entries = []
        for batch in data_loader:
            batch = {k: v.to(config['exp']['device']) for k, v in batch.items()}
            prediction = model(batch)
            unscaled_output = batch['unscaled_outputs']
            prediction = unscale_cancer_data(prediction, config)
            active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(prediction)
            all_predictions.append(prediction)
            all_unscaled_outputs.append(unscaled_output)
            all_active_entries.append(active_entries)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_unscaled_outputs = torch.cat(all_unscaled_outputs, dim=0)
        all_active_entries = torch.cat(all_active_entries, dim=0)
        mses = model.get_mse_at_follow_up_time(all_predictions, all_unscaled_outputs, all_active_entries)
        rmses = torch.sqrt(mses) / 1150 * 100
        
    return rmses, mses

# evaluate the model on sim dataset
def evaluate_sim(model, data_loader, config):
    rmses, mses = evaluate_sim_at_each_follow_up_time(model, data_loader, config)
    rmse = (torch.sqrt(torch.mean(mses)))
    return rmse.item(), [rmses[i].item() for i in range(len(rmses))]

# evaluate the model at each follow-up time on sim dataset
def evaluate_sim_at_each_follow_up_time(model, data_loader, config):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_outputs = []
        for batch in data_loader:
            batch = {k: v.to(config['exp']['device']) for k, v in batch.items()}
            prediction = model(batch)[:,:,0]
            # if config['exp']['autoregressive']:
            #     prediction = prediction[:,:,0]
            output = batch['outputs']
            all_predictions.append(prediction)
            all_outputs.append(output)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_active_entries = batch['active_entries'] if 'active_entries' in batch else torch.ones_like(all_predictions)
        mses = model.get_mse_at_follow_up_time(all_predictions, all_outputs, all_active_entries)
        rmses = torch.sqrt(mses)
        
    return rmses, mses

# evaluate the model
def evaluate(model, data_loader, config):
    if config['dataset']['data'] == 'sim':
        return evaluate_sim(model, data_loader, config)
    elif config['dataset']['data'] == 'cancer':
        return evaluate_cancer(model, data_loader, config)

# log statistics of dataset collection to check if the seed is set correctly
def log_data_seed(dataset_collection, config, logger=None):
    if logger is not None:
        logger.info(f"used seed: {config['exp']['seed']}")
        # log config['dataset']
        # logger.info(f"config['dataset']: {config['dataset']}")
        # logger.info(f"config['model']: {config['model']}")
        # logger.info(f"config['exp']: {config['exp']}")
        logger.info(f'mean of the outputs in the train_f: {np.mean(dataset_collection.train_f.data["unscaled_outputs"])}')
        logger.info(f'mean of the treatments in the train_f: {np.mean(dataset_collection.train_f.data["current_treatments"])}')
        logger.info(f'mean of the covariates in the train_f: {np.mean(dataset_collection.train_f.data["vitals"])}')
        # logger.info(f'mean of the outputs in the test_cf_treatment_seq: {np.mean(dataset_collection.test_cf_treatment_seq.data["outputs"])}')
        
    else:
        print(f"used seed: {config['exp']['seed']}")
        # log config['dataset']
        print(f"config['dataset']: {config['dataset']}")
        print(f'mean of the outputs in the train_f: {np.mean(dataset_collection.train_f.data["outputs"])}')
        # print(f'mean of the outputs in the test_cf_treatment_seq: {np.mean(dataset_collection.test_cf_treatment_seq.data["outputs"])}')

def clear_tfevents(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    # Construct the file path pattern
    file_pattern = os.path.join(directory, '**', 'events.out.tfevents.*')

    # Find files matching the pattern
    files = glob.glob(file_pattern, recursive=True)

    # Delete the found files
    for file_path in files:
        try:
            os.remove(file_path)
            # print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}. Reason: {e}")

def map_values(array):
    mapping_multilable = {
        (0, 0): 0,
        (0, 1): 0.33,
        (1, 0): 0.66,
        (1, 1): 1
    }

    mapping_multiclass = {
        (0, 0, 0, 0): 0.0, # for the case when there is no treatment assigned
        (1, 0, 0, 0): 0.1,
        (0, 1, 0, 0): 0.33,
        (0, 0, 1, 0): 0.66,
        (0, 0, 0, 1): 0.9
        # (0, 0, 0, 0): 0.0, # for the case when there is no treatment assigned
        # (1, 0, 0, 0): 0.,
        # (0, 1, 0, 0): 0.,
        # (0, 0, 1, 0): 0.,
        # (0, 0, 0, 1): 0.
    }
    
    b, T, treatment_size = array.shape
    if treatment_size == 4:
        mapping = mapping_multiclass
    elif treatment_size == 2:
        mapping = mapping_multilable
    else:
        raise ValueError(f'Invalid treatment size: {treatment_size} when mapping values.')

    mapped_array = np.zeros((b, T, 1))

    for i in range(b):
        for j in range(T):
            key = tuple([int(x) for x in array[i, j]])
            # print(f'key: {key}, i: {i}, j: {j}')
            mapped_array[i, j, 0] = mapping[key]
            
    return mapped_array

def add_float_treatment(dataset_collection):
    if hasattr(dataset_collection, 'train_f'):
        dataset_collection.train_f.data['current_treatments_float'] = map_values(dataset_collection.train_f.data['current_treatments'])
        dataset_collection.train_f.data['prev_treatments_float'] = map_values(dataset_collection.train_f.data['prev_treatments'])
    if hasattr(dataset_collection, 'val_f'):
        dataset_collection.val_f.data['current_treatments_float'] = map_values(dataset_collection.val_f.data['current_treatments'])
        dataset_collection.val_f.data['prev_treatments_float'] = map_values(dataset_collection.val_f.data['prev_treatments'])
    if hasattr(dataset_collection, 'test_f'):
        dataset_collection.test_f.data['current_treatments_float'] = map_values(dataset_collection.test_f.data['current_treatments'])
        dataset_collection.test_f.data['prev_treatments_float'] = map_values(dataset_collection.test_f.data['prev_treatments'])
    if hasattr(dataset_collection, 'test_f_multi'):
        dataset_collection.test_f_multi.data['current_treatments_float'] = map_values(dataset_collection.test_f_multi.data['current_treatments'])
        dataset_collection.test_f_multi.data['prev_treatments_float'] = map_values(dataset_collection.test_f_multi.data['prev_treatments'])
        dataset_collection.test_f_multi.data_processed_seq['current_treatments_float'] = map_values(dataset_collection.test_f_multi.data_processed_seq['current_treatments'])
        dataset_collection.test_f_multi.data_processed_seq['prev_treatments_float'] = map_values(dataset_collection.test_f_multi.data_processed_seq['prev_treatments'])
    if hasattr(dataset_collection, 'test_cf'):
        dataset_collection.test_cf.data['current_treatments_float'] = map_values(dataset_collection.test_cf.data['current_treatments'])
        dataset_collection.test_cf.data['prev_treatments_float'] = map_values(dataset_collection.test_cf.data['prev_treatments'])
    if hasattr(dataset_collection, 'test_cf_one_step'):
        dataset_collection.test_cf_one_step.data['current_treatments_float'] = map_values(dataset_collection.test_cf_one_step.data['current_treatments'])
        dataset_collection.test_cf_one_step.data['prev_treatments_float'] = map_values(dataset_collection.test_cf_one_step.data['prev_treatments'])
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
        dataset_collection.test_cf_treatment_seq.data['current_treatments_float'] = map_values(dataset_collection.test_cf_treatment_seq.data['current_treatments'])
        dataset_collection.test_cf_treatment_seq.data['prev_treatments_float'] = map_values(dataset_collection.test_cf_treatment_seq.data['prev_treatments'])
        dataset_collection.test_cf_treatment_seq.data_processed_seq['current_treatments_float'] = map_values(dataset_collection.test_cf_treatment_seq.data_processed_seq['current_treatments'])
        dataset_collection.test_cf_treatment_seq.data_processed_seq['prev_treatments_float'] = map_values(dataset_collection.test_cf_treatment_seq.data_processed_seq['prev_treatments'])

    return dataset_collection

def repeat_static(dataset_collection):
    """
    Repeat static features along the time dimension to match the shape of time-varying features.
    :param dataset_collection:
    :return:
    """
    if hasattr(dataset_collection, 'train_f'):
        static_expanded = np.expand_dims(dataset_collection.train_f.data['static_features'], axis=1)
        dataset_collection.train_f.data['static_features'] = np.repeat(static_expanded, dataset_collection.train_f.data['outputs'].shape[1], axis=1)
    if hasattr(dataset_collection, 'val_f'):
        static_expanded = np.expand_dims(dataset_collection.val_f.data['static_features'], axis=1)
        dataset_collection.val_f.data['static_features'] = np.repeat(static_expanded, dataset_collection.val_f.data['outputs'].shape[1], axis=1)
    if hasattr(dataset_collection, 'test_f'):
        static_expanded = np.expand_dims(dataset_collection.test_f.data['static_features'], axis=1)
        dataset_collection.test_f.data['static_features'] = np.repeat(static_expanded, dataset_collection.test_f.data['outputs'].shape[1], axis=1)
    if hasattr(dataset_collection, 'test_f_multi'):
        static_expanded = np.expand_dims(dataset_collection.test_f_multi.data['static_features'], axis=1)
        dataset_collection.test_f_multi.data['static_features'] = np.repeat(static_expanded, dataset_collection.test_f_multi.data['outputs'].shape[1], axis=1)
        static_expanded = np.expand_dims(dataset_collection.test_f_multi.data_processed_seq['static_features'], axis=1)
        dataset_collection.test_f_multi.data_processed_seq['static_features'] = np.repeat(static_expanded, dataset_collection.test_f_multi.data_processed_seq['outputs'].shape[1], axis=1)
    if hasattr(dataset_collection, 'test_cf'):
        static_expanded = np.expand_dims(dataset_collection.test_cf.data['static_features'], axis=1)
        dataset_collection.test_cf.data['static_features'] = np.repeat(static_expanded, dataset_collection.test_cf.data['outputs'].shape[1], axis=1)
    if hasattr(dataset_collection, 'test_cf_one_step'):
        static_expanded = np.expand_dims(dataset_collection.test_cf_one_step.data['static_features'], axis=1)
        dataset_collection.test_cf_one_step.data['static_features'] = np.repeat(static_expanded, dataset_collection.test_cf_one_step.data['outputs'].shape[1], axis=1)
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
        static_expanded = np.expand_dims(dataset_collection.test_cf_treatment_seq.data['static_features'], axis=1)
        dataset_collection.test_cf_treatment_seq.data['static_features'] = np.repeat(static_expanded, dataset_collection.test_cf_treatment_seq.data['outputs'].shape[1], axis=1)
        static_expanded = np.expand_dims(dataset_collection.test_cf_treatment_seq.data_processed_seq['static_features'], axis=1)
        dataset_collection.test_cf_treatment_seq.data_processed_seq['static_features'] = np.repeat(static_expanded, dataset_collection.test_cf_treatment_seq.data_processed_seq['outputs'].shape[1], axis=1)

    return dataset_collection

def del_static(dataset_collection):
    if hasattr(dataset_collection, 'train_f'):
        static_expanded = np.expand_dims(dataset_collection.train_f.data['static_features'], axis=1)
        dataset_collection.train_f.data['static_features'] = None
    if hasattr(dataset_collection, 'val_f'):
        static_expanded = np.expand_dims(dataset_collection.val_f.data['static_features'], axis=1)
        dataset_collection.val_f.data['static_features'] = None
    if hasattr(dataset_collection, 'test_f'):
        static_expanded = np.expand_dims(dataset_collection.test_f.data['static_features'], axis=1)
        dataset_collection.test_f.data['static_features'] = None
    if hasattr(dataset_collection, 'test_f_multi'):
        static_expanded = np.expand_dims(dataset_collection.test_f_multi.data['static_features'], axis=1)
        dataset_collection.test_f_multi.data['static_features'] = None
        static_expanded = np.expand_dims(dataset_collection.test_f_multi.data_processed_seq['static_features'], axis=1)
        dataset_collection.test_f_multi.data_processed_seq['static_features'] = None
    if hasattr(dataset_collection, 'test_cf'):
        static_expanded = np.expand_dims(dataset_collection.test_cf.data['static_features'], axis=1)
        dataset_collection.test_cf.data['static_features'] = None
    if hasattr(dataset_collection, 'test_cf_one_step'):
        static_expanded = np.expand_dims(dataset_collection.test_cf_one_step.data['static_features'], axis=1)
        dataset_collection.test_cf_one_step.data['static_features'] = None
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
        static_expanded = np.expand_dims(dataset_collection.test_cf_treatment_seq.data['static_features'], axis=1)
        dataset_collection.test_cf_treatment_seq.data['static_features'] = None
        static_expanded = np.expand_dims(dataset_collection.test_cf_treatment_seq.data_processed_seq['static_features'], axis=1)
        dataset_collection.test_cf_treatment_seq.data_processed_seq['static_features'] = None

    return dataset_collection

def to_float(dataset_collection):
    if hasattr(dataset_collection, 'train_f'):
        for key in dataset_collection.train_f.data.keys():
            dataset_collection.train_f.data[key] = dataset_collection.train_f.data[key].astype(np.float32)
    if hasattr(dataset_collection, 'val_f'):
        for key in dataset_collection.val_f.data.keys():
            dataset_collection.val_f.data[key] = dataset_collection.val_f.data[key].astype(np.float32)
    if hasattr(dataset_collection, 'test_f'):
        for key in dataset_collection.test_f.data.keys():
            dataset_collection.test_f.data[key] = dataset_collection.test_f.data[key].astype(np.float32)
    if hasattr(dataset_collection, 'test_f_multi'):
        for key in dataset_collection.test_f_multi.data.keys():
            dataset_collection.test_f_multi.data[key] = dataset_collection.test_f_multi.data[key].astype(np.float32)
        for key in dataset_collection.test_f_multi.data_processed_seq.keys():
            dataset_collection.test_f_multi.data_processed_seq[key] = dataset_collection.test_f_multi.data_processed_seq[key].astype(np.float32)
    if hasattr(dataset_collection, 'test_cf'):
        for key in dataset_collection.test_cf.data.keys():
            dataset_collection.test_cf.data[key] = dataset_collection.test_cf.data[key].astype(np.float32)
    if hasattr(dataset_collection, 'test_cf_one_step'):
        for key in dataset_collection.test_cf_one_step.data.keys():
            dataset_collection.test_cf_one_step.data[key] = dataset_collection.test_cf_one_step.data[key].astype(np.float32)
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
        for key in dataset_collection.test_cf_treatment_seq.data.keys():
            dataset_collection.test_cf_treatment_seq.data[key] = dataset_collection.test_cf_treatment_seq.data[key].astype(np.float32)
        try:
            for key in dataset_collection.test_cf_treatment_seq.data_processed_seq.keys():
                dataset_collection.test_cf_treatment_seq.data_processed_seq[key] = dataset_collection.test_cf_treatment_seq.data_processed_seq[key].astype(np.float32)
        except:
            pass
    
    return dataset_collection

def to_double(dataset_collection):
    if hasattr(dataset_collection, 'train_f'):
        for key in dataset_collection.train_f.data.keys():
            dataset_collection.train_f.data[key] = dataset_collection.train_f.data[key].astype(np.float64)
    if hasattr(dataset_collection, 'val_f'):
        for key in dataset_collection.val_f.data.keys():
            dataset_collection.val_f.data[key] = dataset_collection.val_f.data[key].astype(np.float64)
    if hasattr(dataset_collection, 'test_f'):
        for key in dataset_collection.test_f.data.keys():
            dataset_collection.test_f.data[key] = dataset_collection.test_f.data[key].astype(np.float64)
    if hasattr(dataset_collection, 'test_f_multi'):
        for key in dataset_collection.test_f_multi.data.keys():
            dataset_collection.test_f_multi.data[key] = dataset_collection.test_f_multi.data[key].astype(np.float64)
        for key in dataset_collection.test_f_multi.data_processed_seq.keys():
            dataset_collection.test_f_multi.data_processed_seq[key] = dataset_collection.test_f_multi.data_processed_seq[key].astype(np.float64)
    if hasattr(dataset_collection, 'test_cf'):
        for key in dataset_collection.test_cf.data.keys():
            dataset_collection.test_cf.data[key] = dataset_collection.test_cf.data[key].astype(np.float64)
    if hasattr(dataset_collection, 'test_cf_one_step'):
        for key in dataset_collection.test_cf_one_step.data.keys():
            dataset_collection.test_cf_one_step.data[key] = dataset_collection.test_cf_one_step.data[key].astype(np.float64)
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
        for key in dataset_collection.test_cf_treatment_seq.data.keys():
            dataset_collection.test_cf_treatment_seq.data[key] = dataset_collection.test_cf_treatment_seq.data[key].astype(np.float64)
        try:
            for key in dataset_collection.test_cf_treatment_seq.data_processed_seq.keys():
                dataset_collection.test_cf_treatment_seq.data_processed_seq[key] = dataset_collection.test_cf_treatment_seq.data_processed_seq[key].astype(np.float64)
        except:
            pass
    return dataset_collection

def count_parameters(model, logger=None):
    total_params = 0
    for name, parameter in model.named_parameters():
        # Skip parameters that are not trainable
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        if logger:
            logger.info(f"{name}: {param}")
        else:

            print(f"{name}: {param}")
    if logger:
        logger.info(f"Total Trainable Params: {total_params}")
    else:
        print(f"Total Trainable Params: {total_params}")

class AlphaRise(Callback):
    """
    Exponential lambda_D rise
    """
    def __init__(self, rate='exp'):
        self.rate = rate

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if pl_module.update_lambda_D:
            p = float(pl_module.current_epoch + 1) / float(pl_module.epochs)
            if self.rate == 'lin':
                pl_module.lambda_D = p * pl_module.lambda_D_max
            elif self.rate == 'exp':
                pl_module.lambda_D = (2. / (1. + np.exp(-10. * p)) - 1.0) * pl_module.lambda_D_max
            else:
                raise NotImplementedError()

def rbf_kernel(x, y, sigma=1.0):
    """
    Computes the Radial Basis Function (RBF) kernel between two tensors.
    """
    beta = 1. / (2. * sigma ** 2)
    dist_matrix = torch.cdist(x, y) ** 2
    return torch.exp(-beta * dist_matrix)

def compute_mmd(x, y, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) distance between two samples: x and y.
    The MMD is a distance measure between the samples of the distributions P and Q.
    """
    x_kernel = rbf_kernel(x, x, sigma)
    y_kernel = rbf_kernel(y, y, sigma)
    xy_kernel = rbf_kernel(x, y, sigma)
    
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def compute_mmd_weighted(
    joint_rep: torch.Tensor,
    marginal_rep: torch.Tensor,
    w_t: torch.Tensor,
    sigma: float = 1.0,
    sigmas: Optional[Sequence[float]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Weighted MMD between a **weighted** empirical law on ``joint_rep`` and a **uniform**
    empirical law on ``marginal_rep``.

    Matches the CTD-style deconfounding objective (see ``.cursor/skills/ctd.md``): align
    the reweighted joint :math:`P_w(Z_t,A_t)` with the product-of-marginals surrogate built
    by shuffling :math:`A_t`, i.e. samples ``marginal_rep = cat(Z_t, A_t[perm])``.

    **Difference vs local CTD-NKO_IJCAI code:** that repository does **not** implement MMD for
    this step. It uses ``geomloss.SamplesLoss`` **Sinkhorn** in
    ``CTD-NKO_IJCAI-main/src/models/wass_calculator.py`` (:func:`compute_weighted_wasserstein_geomloss`).
    For numerical parity with the paper code, use that Sinkhorn loss; use this function when you
    want an MMD-based alignment (as allowed by ``ctd.md`` wording).

    Uses the same RBF kernel as :func:`rbf_kernel` / ``generative_model.mmd_loss`` (multi-scale
    optional, as in common MMD-GAN / representation balancing code).

    Args:
        joint_rep: ``[N, D]`` concatenated :math:`[Z_t, A_t]` (or any joint features).
        marginal_rep: ``[N, D]`` shuffled-marginal features, same batch size ``N``.
        w_t: ``[N]`` or ``[N, 1]`` non-negative weights (e.g. from WeightNet). Internally
            normalized to sum to 1 so they act as a probability vector over batch points.
        sigma: Base RBF bandwidth if ``sigmas`` is None.
        sigmas: If set, average MMD over these bandwidths (same spirit as multi-σ in
            ``mmd_loss`` in ``generative_model.py``).
        eps: Small constant for weight normalization stability.

    Returns:
        Scalar tensor, mean MMD^2 over chosen bandwidths (biased estimator, diagonals included,
        consistent with :func:`compute_mmd`).
    """
    if joint_rep.shape != marginal_rep.shape:
        raise ValueError(
            f"joint_rep and marginal_rep must have the same shape, got "
            f"{tuple(joint_rep.shape)} vs {tuple(marginal_rep.shape)}"
        )
    n = joint_rep.size(0)
    if n == 0:
        return joint_rep.new_zeros(())

    w = w_t.reshape(-1).to(dtype=joint_rep.dtype, device=joint_rep.device)
    if w.numel() != n:
        raise ValueError(f"w_t must have length N={n}, got {w.numel()}")
    alpha = w / (w.sum() + eps)

    bandwidths: Union[list, tuple]
    if sigmas is not None:
        bandwidths = tuple(float(s) for s in sigmas)
    else:
        bandwidths = (float(sigma),)

    mmd_acc = joint_rep.new_zeros(())
    for s in bandwidths:
        k_xx = rbf_kernel(joint_rep, joint_rep, s)
        k_yy = rbf_kernel(marginal_rep, marginal_rep, s)
        k_xy = rbf_kernel(joint_rep, marginal_rep, s)
        # Weighted joint–joint term: α^T K_xx α
        term_xx = torch.einsum("i,ij,j->", alpha, k_xx, alpha)
        # Uniform on marginal sample: E_{y,y'~Unif}[k(y,y')] = mean of K_yy
        term_yy = k_yy.mean()
        # Cross term: E_{x~α,y~Unif}[k(x,y)] = (1/N) Σ_j Σ_i α_i K_xy[i,j]
        uniform = torch.full((n,), 1.0 / n, device=joint_rep.device, dtype=joint_rep.dtype)
        term_xy = torch.einsum("i,ij,j->", alpha, k_xy, uniform)
        mmd_acc = mmd_acc + (term_xx + term_yy - 2.0 * term_xy)

    return mmd_acc / float(len(bandwidths))


def compute_weighted_wasserstein_geomloss(A, B, wa, wb=None, blur: float = 0.01, p: int = 2):
    """
    Aligned with ``CTD-NKO_IJCAI-main/src/models/wass_calculator.compute_weighted_wasserstein_geomloss``.

    Weighted Sinkhorn / Wasserstein-type loss between samples ``A`` and ``B`` using ``geomloss``.
    ``wa`` are softmax-normalized on batch dimension (dim 0); if ``wb`` is omitted, ``B`` uses
    uniform weights (softmax of ones on dim 0), matching the reference implementation.

    Args:
        A: ``(n, T, dim)`` joint (or other) features per sequence timestep.
        B: ``(n, T, dim)`` marginal / shuffled counterpart, same shape as ``A``.
        wa: ``(n, T, 1)`` raw weights (e.g. sigmoid from a weight net); internally ``F.softmax(..., dim=0)``.
        wb: optional ``(n, T, 1)``; default uniform over ``n`` at each ``t``.
        blur: Sinkhorn entropic temperature (CTD-NKO uses ``0.01``).
        p: ground cost exponent (CTD-NKO uses ``2``).

    Returns:
        Scalar: mean over time of per-step Sinkhorn losses.
    """
    try:
        from geomloss import SamplesLoss
    except ImportError as err:
        raise ImportError(
            "compute_weighted_wasserstein_geomloss requires the `geomloss` package "
            "(see CTD-NKO_IJCAI requirements.txt)."
        ) from err

    if wb is None:
        wb = torch.ones_like(wa).to(wa.device)
        wb = F.softmax(wb, dim=0)
    wa = F.softmax(wa, dim=0)
    n, T, dim = A.shape
    distances = torch.zeros(T, 1, device=A.device, dtype=A.dtype)
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    for t in range(T):
        distances[t] = loss(wa[:, t], A[:, t], wb[:, t], B[:, t])
    return distances.mean()


def compute_weighted_wasserstein_joint_marginal_flat(
    joint_rep: torch.Tensor,
    marginal_rep: torch.Tensor,
    w_t: torch.Tensor,
    blur: float = 0.01,
) -> torch.Tensor:
    """
    Same distribution-alignment **objective** as CTD-NKO, for flat batch tensors ``[N, D]``
    (e.g. one time index per row). Wraps :func:`compute_weighted_wasserstein_geomloss` with
    ``T=1``.

    CTD-NKO builds ``real_samples`` / ``fake_samples`` as ``[n, T, dim]``; this helper is for
    ``train_ct``-style batches where each row is already ``concat([Z_t, A_t], dim=-1)``.
    """
    if joint_rep.shape != marginal_rep.shape:
        raise ValueError(
            f"joint_rep and marginal_rep must have the same shape, got "
            f"{tuple(joint_rep.shape)} vs {tuple(marginal_rep.shape)}"
        )
    n = joint_rep.size(0)
    w = w_t.reshape(n, 1, 1).to(device=joint_rep.device, dtype=joint_rep.dtype)
    A = joint_rep.unsqueeze(1)
    B = marginal_rep.unsqueeze(1)
    return compute_weighted_wasserstein_geomloss(A, B, w, blur=blur)


def pdist2sq(A, B):
    na = torch.sum(torch.square(A), 1)
    nb = torch.sum(torch.square(B), 1)  
    # na as a row and nb as a column vectors
    na = torch.reshape(na, [-1, 1])
    nb = torch.reshape(nb, [1, -1])
    # return pairwise euclidean difference matrix
    D = torch.sum((torch.unsqueeze(A, 1) - torch.unsqueeze(B, 0))**2, 2) 
    return D

def rbf_kernel_v2(A, B, rbf_sigma=1):
    rbf_sigma = torch.tensor(rbf_sigma)
    return torch.exp(-pdist2sq(A, B) / torch.square(rbf_sigma) *.5)

def calculate_mmd(A, B, rbf_sigma=1):
    Kaa = rbf_kernel_v2(A, A, rbf_sigma)
    Kab = rbf_kernel_v2(A, B, rbf_sigma)
    Kbb = rbf_kernel_v2(B, B, rbf_sigma)
    m = A.shape[0]
    n = B.shape[0]
    # print(m, n)
    mmd = 1.0 / (m * (m - 1.0)) * (torch.sum(Kaa) - m)
    mmd = mmd - 2.0 / (m * n) * torch.sum(Kab)
    mmd = mmd + 1.0 / (n * (n - 1.0)) * (torch.sum(Kbb) - n)

    return mmd

def eval(model, dataset_collection):
    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(dataset_collection.val_f)
    results = {}
    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = model.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
                                                                                              one_step_counterfactual=True)
        encoder_results = {
            'val_rmse_all': val_rmse_all,
            'test_rmse_all': test_rmse_all,
            'test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(dataset_collection.test_f)
        encoder_results = {
            'val_rmse_all': val_rmse_all,
            'test_rmse_all': test_rmse_all,
        }
    results.update(encoder_results)
    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}
    results.update(test_rmses)
    return results

def advanced_indexing_shuffle_3d(a, mode='all'):
    n, seq, dim = a.shape
    device = a.device 
    if mode == 'all':
        idx = torch.stack([torch.randperm(n) for _ in range(seq)]).T
        return a[idx, torch.arange(seq), :]
    else:
        idx = torch.randperm(n)
        return a[idx, :, :]
    # idx = torch.stack([torch.randperm(n) for _ in range(seq)]).T
    # return a[idx, torch.arange(seq), :]

def sample_treatments(treatments, mode):
    n, seq, _ = treatments.shape
    if mode == 'multilabel':
        random_treatments = torch.randint(0, 2, (n, seq, 2))
    elif mode == 'multiclass':
        indices = torch.randint(0, 4, (n, seq))
        random_treatments = torch.nn.functional.one_hot(indices, num_classes=4)
    else:
        raise ValueError(f'Invalid mode: {mode}')
    
    return random_treatments


def get_conv_max_weights(net):
    max_weights = []
    for temporal_block in net.network:
        for conv in [temporal_block.conv1, temporal_block.conv2]:
            max_weight = conv.weight.data.max().item()  # 获取当前卷积核权重的最大值
            max_weights.append(max_weight)
    return max_weights.max()