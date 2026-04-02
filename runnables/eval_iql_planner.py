"""
Evaluate IQL planner with the same simulator and RMSE metric as VAEModel.optimize_interventions_onetime:
  output_after_actions = fold.simulate_output_after_actions(H_t, a_seq, train_scaling_params)
  RMSE on normalized cancer_volume vs targets['outputs'][:, -1, :], then × std (cancer) like VCIP.
"""
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.iql_dataset_builder import align_h_t_static_to_history
from src.models.inference_model import InferenceModel
from src.planners.iql_planner import IQLPlanner
from src.utils.inference_ckpt import load_inference_checkpoint
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _actions_to_sim_interval(raw: np.ndarray, max_action: float) -> np.ndarray:
    """Map actor output in [-max_action, max_action] to [0, 1] for cancer chemo/radio sim."""
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return np.clip((raw + max_action) / denom, 0.0, 1.0).astype(np.float32)


def _resolve_iql_ckpt(args: DictConfig, original_cwd: Path) -> Path:
    explicit = OmegaConf.select(args, "exp.iql_eval_ckpt", default="")
    if explicit:
        p = Path(str(explicit))
        if not p.is_absolute():
            p = original_cwd / p
        return p
    seed = int(args.exp.seed)
    gamma = int(args.dataset.coeff)
    return original_cwd / "iql_models" / f"seed_{seed}" / f"gamma_{gamma}" / "iql_planner.pt"


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(args.exp.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    # 这一行代码的作用是：根据配置文件 `args.dataset` 所指定的配置参数，实例化（创建）一个对象（比如数据集的集合/管理类）。
    # `instantiate` 是 Hydra 框架里的一个辅助函数，它可以根据配置字典（比如 OmegaConf 风格的嵌套 dict）自动初始化相应 Python 类/对象。常用于机器学习工程的配置驱动式组件构建。
    # 参数 `_recursive_=True` 表示如果配置中有嵌套的配置项，也会递归地实例化嵌套项。比如 dataset 里定义了什么类、有什么参数、嵌套子对象等，都能自动注册并创建出来。
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    try:
        std = float(dataset_collection.train_scaling_params[1]["cancer_volume"])
    except Exception:
        std = 1.0

    if args.exp.test:
        data = dataset_collection.test_f.data
        fold = dataset_collection.test_f
        split_name = "test"
    else:
        data = dataset_collection.val_f.data
        fold = dataset_collection.val_f
        split_name = "val"

    batch_size = int(OmegaConf.select(args, "exp.batch_size_val", default=128))
    dataloader = get_dataloader(CIPDataset(data, args, train=False), batch_size=batch_size, shuffle=False)

    iql_ckpt = str(OmegaConf.select(args, "exp.iql_inference_ckpt", default=""))
    inference_model = InferenceModel(args).to(device)
    load_inference_checkpoint(inference_model, iql_ckpt, device)
    inference_model.eval()

    planner_path = _resolve_iql_ckpt(args, original_cwd)
    if not planner_path.exists():
        raise FileNotFoundError(f"IQL checkpoint not found: {planner_path}. Set exp.iql_eval_ckpt or train first.")
    planner = IQLPlanner.from_checkpoint(str(planner_path), device=device)
    max_action = float(planner.cfg.max_action)
    tau = int(args.exp.tau)

    losses = []
    losses_2 = []
    ture_output_list = []
    output_after_actions_list = []
    ture_output_actions_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            H_t, targets = batch
            H_t = align_h_t_static_to_history(H_t)
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            z, _, _ = inference_model.ct_hidden_history(H_t)
            z_np = z.detach().cpu().numpy()
            bsz = z_np.shape[0]
            a_rows = []
            for b in range(bsz):
                a_rows.append(planner.act(z_np[b]))
            a_raw = np.stack(a_rows, axis=0)
            a_sim = _actions_to_sim_interval(a_raw, max_action)
            a_seq = torch.tensor(a_sim, device=device, dtype=torch.float32).unsqueeze(1).expand(-1, tau, -1).contiguous()

            output_after_actions = fold.simulate_output_after_actions(
                H_t, a_seq, dataset_collection.train_scaling_params
            )
            ture_output = targets["outputs"][:, -1, :].detach().cpu().numpy()
            loss = np.sqrt(((output_after_actions - ture_output) ** 2).mean())
            losses.append(loss)

            true_actions = targets["current_treatments"]
            ture_output_actions = fold.simulate_output_after_actions(
                H_t, true_actions, dataset_collection.train_scaling_params
            )
            loss_2 = np.sqrt(((ture_output_actions - ture_output) ** 2).mean())
            losses_2.append(loss_2)

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)
            ture_output_actions_list.append(ture_output_actions)

            logger.info(f"Batch {i} RMSE (IQL plan): {loss:.6f}, RMSE (factual actions): {loss_2:.6f}")

    ture_output_list = np.concatenate(ture_output_list, axis=0)
    output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)
    ture_output_actions_list = np.concatenate(ture_output_actions_list, axis=0)

    rmse_norm = float(np.sqrt(((output_after_actions_list - ture_output_list) ** 2).mean()))
    rmse_factual_norm = float(np.sqrt(((ture_output_actions_list - ture_output_list) ** 2).mean()))

    logger.info("--- Aggregate (same protocol as optimize_interventions_onetime) ---")
    logger.info(f"Split: {split_name}")
    logger.info(f"Mean per-batch RMSE (IQL): {float(np.mean(losses)):.6f}")
    logger.info(f"Mean per-batch RMSE (factual traj): {float(np.mean(losses_2)):.6f}")
    logger.info(f"Global RMSE on stacked batches (normalized space): {rmse_norm:.6f}")
    logger.info(f"Global RMSE × std (cancer volume scale, VCIP-style): {rmse_norm * std:.6f}")
    logger.info(f"Factual global RMSE × std: {rmse_factual_norm * std:.6f}")


if __name__ == "__main__":
    main()
