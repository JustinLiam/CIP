"""
Evaluate IQL planner with the same simulator and RMSE metric as VAEModel.optimize_interventions_onetime:
  output_after_actions = fold.simulate_output_after_actions(H_t, a_seq, train_scaling_params)
  RMSE on normalized cancer_volume vs targets['outputs'][:, -1, :], then × std (cancer) like VCIP.

By default (``exp.iql_eval_autoregressive=true``) the planned sequence is built autoregressively: each step
re-encodes with ``ct_hidden_history`` after appending the chosen action and one-step simulated outcome.
The final RMSE still uses the original prefix ``H_t`` and the full ``a_seq`` (same closed-loop sim as VCIP).

Logs aggregate **t+tau** tumor volume under the IQL plan vs ``targets['outputs'][:, -1]``, in normalized
(train scaling) and unscaled (raw simulator scale) space.
"""
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.distributions import Distribution
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
    """
    Inverse of ``dataset_actions_to_tanh_policy_space``: map Tanh policy output in
    [-max_action, max_action] to simulator treatment interval [0, max_action] (typically [0, 1]).
    """
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return np.clip((raw + max_action) / denom, 0.0, 1.0).astype(np.float32)


def _policy_to_sim_interval_torch(raw: torch.Tensor, max_action: float) -> torch.Tensor:
    """Batched tensor version of ``_actions_to_sim_interval``."""
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return torch.clamp((raw + max_action) / denom, 0.0, 1.0)


def _unscaled_cancer_volume(y_norm: torch.Tensor, mean_ser, std_ser) -> torch.Tensor:
    """y_norm: [B, 1] normalized tumor volume -> unscaled [B, 1]."""
    m = float(mean_ser["cancer_volume"])
    s = float(std_ser["cancer_volume"])
    return y_norm * s + m


def _unscaled_cancer_volume_np(y_norm: np.ndarray, mean_ser, std_ser) -> np.ndarray:
    """Same as ``_unscaled_cancer_volume`` for numpy (e.g. simulator outputs)."""
    m = float(mean_ser["cancer_volume"])
    s = float(std_ser["cancer_volume"])
    return y_norm.astype(np.float64) * s + m


def _extend_h_work_after_one_step(
    H: dict,
    a_sim: torch.Tensor,
    y_norm: torch.Tensor,
    mean_ser,
    std_ser,
    device: torch.device,
) -> None:
    """
    In-place: append one planned (chemo, radio) step and simulated outcome so the next
    ``ct_hidden_history`` sees an updated prefix (autoregressive planning).

    Updates simulator-facing arrays (cancer_volume, unscaled_outputs, chemo/radio) and
    encoder-facing streams (prev/current treatments, outputs, prev_outputs, active_entries,
    static_features, current_covariates when present).
    """
    B = a_sim.size(0)
    y_col = y_norm.view(B, 1)
    y_ch = y_col.unsqueeze(-1)  # [B, 1, 1] for outputs
    y_uns = _unscaled_cancer_volume(y_col, mean_ser, std_ser)

    last_curr = H["current_treatments"][:, -1:, :].clone()
    last_out = H["outputs"][:, -1:, :].clone()

    H["prev_treatments"] = torch.cat([H["prev_treatments"], last_curr], dim=1)
    H["current_treatments"] = torch.cat([H["current_treatments"], a_sim.unsqueeze(1)], dim=1)
    H["outputs"] = torch.cat([H["outputs"], y_ch], dim=1)
    H["prev_outputs"] = torch.cat([H["prev_outputs"], last_out], dim=1)
    ae = H["active_entries"]
    H["active_entries"] = torch.cat(
        [ae, torch.ones(B, 1, ae.size(-1), device=device, dtype=ae.dtype)], dim=1
    )

    H["cancer_volume"] = torch.cat([H["cancer_volume"], y_col], dim=1)

    uo = H["unscaled_outputs"]
    y_u = y_uns.unsqueeze(-1) if uo.dim() == 3 else y_uns
    H["unscaled_outputs"] = torch.cat([uo, y_u], dim=1)

    H["chemo_application"] = torch.cat([H["chemo_application"], a_sim[:, 0:1]], dim=1)
    H["radio_application"] = torch.cat([H["radio_application"], a_sim[:, 1:2]], dim=1)

    if "static_features" in H:
        sf = H["static_features"]
        if sf.dim() == 3:
            last = sf[:, -1:, :].expand(-1, 1, -1)
            H["static_features"] = torch.cat([sf, last], dim=1)

    if "current_covariates" in H:
        cc = H["current_covariates"]
        ext = cc[:, -1:, :].clone()
        ext[:, :, 0:1] = y_ch
        H["current_covariates"] = torch.cat([cc, ext], dim=1)


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

    # args.exp.test 参数用于指定当前代码是在“测试集” (test set) 上运行，还是在“验证集” (validation set) 上运行。
    # 如果 args.exp.test 为 True，则采用测试集数据 dataset_collection.test_f 进行评估，split_name 为 "test"。
    # 如果为 False，则采用验证集数据 dataset_collection.val_f，split_name 为 "val"。
    # 这样可以通过配置文件或者命令行参数 exp.test=true/false 灵活切换评估集。
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
    # tau 加载自 args.exp.tau，默认由 cancer_sim_cont.yaml（如 dataset/projection_horizon/tau）或 config.yaml（如 exp.tau）提供。
    # 你可以在命令行用 `python runnables/eval_iql_planner.py exp.tau=8` 这样的覆盖参数，Hydra 会自动覆盖 config.yaml 和任何 defaults 里的 tau 配置。
    tau = int(args.exp.tau)
    autoregressive_eval = bool(OmegaConf.select(args, "exp.iql_eval_autoregressive", default=True))
    mean_ser, std_ser = dataset_collection.train_scaling_params
    logger.info(f"IQL eval autoregressive action rollout: {autoregressive_eval} (tau={tau})")

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

            if autoregressive_eval:
                H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                planned = []
                for _ in range(tau):
                    H_work = align_h_t_static_to_history(H_work)
                    z, _, _ = inference_model.ct_hidden_history(H_work)
                    po = planner.actor(z)
                    ma = planner.actor.max_action
                    if isinstance(po, Distribution):
                        a_raw = torch.clamp(ma * po.mean, -ma, ma)
                    else:
                        a_raw = torch.clamp(po * ma, -ma, ma)
                    a_sim = _policy_to_sim_interval_torch(a_raw, max_action)
                    planned.append(a_sim)
                    y_np = fold.simulate_output_after_actions(
                        H_work,
                        a_sim.unsqueeze(1),
                        dataset_collection.train_scaling_params,
                    )
                    y_norm = torch.as_tensor(y_np, device=device, dtype=torch.float32)
                    _extend_h_work_after_one_step(
                        H_work, a_sim, y_norm, mean_ser, std_ser, torch.device(device)
                    )
                a_seq = torch.stack(planned, dim=1).contiguous()
            else:
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

    # t+tau tumor volume under IQL plan vs ground truth (normalized = train scaling space)
    iql_y_norm = output_after_actions_list.reshape(-1)
    true_y_norm = ture_output_list.reshape(-1)
    iql_y_uns = _unscaled_cancer_volume_np(output_after_actions_list, mean_ser, std_ser).reshape(-1)
    true_y_uns = _unscaled_cancer_volume_np(ture_output_list, mean_ser, std_ser).reshape(-1)

    mae_norm = float(np.mean(np.abs(iql_y_norm - true_y_norm)))
    mae_uns = float(np.mean(np.abs(iql_y_uns - true_y_uns)))
    rmse_uns = float(np.sqrt(np.mean((iql_y_uns - true_y_uns) ** 2)))

    logger.info("--- Aggregate (same protocol as optimize_interventions_onetime) ---")
    logger.info(f"Split: {split_name}")
    logger.info(f"Mean per-batch RMSE (IQL): {float(np.mean(losses)):.6f}")
    logger.info(f"Mean per-batch RMSE (factual traj): {float(np.mean(losses_2)):.6f}")
    logger.info(f"Global RMSE on stacked batches (normalized space): {rmse_norm:.6f}")
    logger.info(f"Global RMSE × std (cancer volume scale, VCIP-style): {rmse_norm * std:.6f}")
    logger.info(f"Factual global RMSE × std: {rmse_factual_norm * std:.6f}")

    logger.info("--- t+tau tumor volume (IQL planned actions vs target outputs[:, -1]) ---")
    logger.info(
        f"IQL pred normalized:   mean={float(np.mean(iql_y_norm)):.6f} std={float(np.std(iql_y_norm)):.6f} "
        f"min={float(np.min(iql_y_norm)):.6f} max={float(np.max(iql_y_norm)):.6f}"
    )
    logger.info(
        f"Target normalized:       mean={float(np.mean(true_y_norm)):.6f} std={float(np.std(true_y_norm)):.6f} "
        f"min={float(np.min(true_y_norm)):.6f} max={float(np.max(true_y_norm)):.6f}"
    )
    logger.info(
        f"IQL pred unscaled:     mean={float(np.mean(iql_y_uns)):.6f} std={float(np.std(iql_y_uns)):.6f} "
        f"min={float(np.min(iql_y_uns)):.6f} max={float(np.max(iql_y_uns)):.6f}"
    )
    logger.info(
        f"Target unscaled:       mean={float(np.mean(true_y_uns)):.6f} std={float(np.std(true_y_uns)):.6f} "
        f"min={float(np.min(true_y_uns)):.6f} max={float(np.max(true_y_uns)):.6f}"
    )
    logger.info(
        f"Train scaling cancer_volume: mean={float(mean_ser['cancer_volume']):.6f} std={float(std_ser['cancer_volume']):.6f}"
    )
    logger.info(f"MAE normalized: {mae_norm:.6f} | MAE unscaled: {mae_uns:.6f} | RMSE unscaled: {rmse_uns:.6f}")


if __name__ == "__main__":
    main()
