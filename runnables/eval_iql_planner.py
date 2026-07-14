"""
Evaluate the IQL planner as an online closed-loop policy.

The main metric is RMSE between the final y_norm observed during autoregressive
closed-loop rollout and targets["outputs"][:, -1, :], reported in normalized space
and unscaled tumor-volume space.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cip_dataset import CIPDataset, get_dataloader, planning_repeats
from src.data.iql_dataset_builder import align_h_t_static_to_history, dataset_actions_to_tanh_policy_space
from src.evaluation.iql_planner_eval import aggregate_iql_planner_metrics
from src.evaluation.iql_action_selection import select_iql_policy_action
from src.models.inference_model import InferenceModel
from src.models.sequence_utils import gather_last_valid
from src.planners.iql_planner import IQLPlanner
from src.utils.em_ckpt import is_em_checkpoint, load_em_for_eval
from src.utils.inference_ckpt import load_inference_checkpoint
from src.utils.mlflow_vcip import VCIPMlflowTracker
from src.utils.stable_iql_em_defaults import stable_select
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)



def _actions_to_sim_interval(raw: np.ndarray, max_action: float) -> np.ndarray:
    """
    Inverse of ``dataset_actions_to_tanh_policy_space``: map Tanh policy output in
    [-max_action, max_action] to simulator treatment interval [0, 1].
    """
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return np.clip((raw + max_action) / denom, 0.0, 1.0).astype(np.float32)


def _policy_to_sim_interval_torch(raw: torch.Tensor, max_action: float) -> torch.Tensor:
    """Batched tensor version of ``_actions_to_sim_interval``."""
    denom = 2.0 * max_action if max_action > 0 else 1.0
    return torch.clamp((raw + max_action) / denom, 0.0, 1.0)


def _calibrate_sim_actions_torch(a_sim: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    """Optional eval-time action calibration in simulator [0, 1] space."""
    if scale == 1.0 and shift == 0.0:
        return a_sim
    return torch.clamp(a_sim * float(scale) + float(shift), 0.0, 1.0)


def _calibrate_sim_actions_np(a_sim: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """Numpy equivalent of ``_calibrate_sim_actions_torch``."""
    if scale == 1.0 and shift == 0.0:
        return a_sim
    return np.clip(a_sim * float(scale) + float(shift), 0.0, 1.0).astype(np.float32)


def _sim_actions_to_tanh_batch(a_sim: torch.Tensor, max_action: float) -> torch.Tensor:
    """Match ``dataset_actions_to_tanh_policy_space`` for batched simulator actions [B, A]."""
    if max_action <= 0:
        return a_sim
    a = torch.clamp(a_sim, 0.0, 1.0)
    return (2.0 * a - 1.0) * float(max_action)


def _iql_augmented_state(
    planner: IQLPlanner,
    z: torch.Tensor,
    eval_target: torch.Tensor,
    step: int,
    eval_tau: int,
    max_tau: float,
    a_prev_tanh: torch.Tensor,
) -> torch.Tensor:
    """concat(Z_t, Y_goal, delta_t_norm, a_{t-1}) in policy space; delta_t_norm = (eval_tau - step) / max_tau."""
    bsz = z.size(0)
    steps_left = float(eval_tau - step)
    delta = torch.full((bsz, 1), steps_left / max_tau, device=z.device, dtype=z.dtype)
    return planner.build_state(z, eval_target, delta, a_prev_tanh)


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


def _split_scaling_params(scaling_params):
    if isinstance(scaling_params, dict) and "output_means" in scaling_params:
        means = np.asarray(scaling_params["output_means"], dtype=np.float64).reshape(1, -1)
        stds = np.asarray(scaling_params["output_stds"], dtype=np.float64).reshape(1, -1)
        return means, stds
    mean_ser, std_ser = scaling_params
    if hasattr(mean_ser, "__getitem__") and "cancer_volume" in mean_ser:
        return (
            np.asarray([[float(mean_ser["cancer_volume"])]], dtype=np.float64),
            np.asarray([[float(std_ser["cancer_volume"])]], dtype=np.float64),
        )
    return (
        np.asarray(mean_ser, dtype=np.float64).reshape(1, -1),
        np.asarray(std_ser, dtype=np.float64).reshape(1, -1),
    )


def _unscale_outputs_np(y_norm: np.ndarray, scaling_params) -> np.ndarray:
    means, stds = _split_scaling_params(scaling_params)
    return np.asarray(y_norm, dtype=np.float64) * stds + means


def _unscale_outputs_torch(y_norm: torch.Tensor, scaling_params, device: torch.device) -> torch.Tensor:
    means, stds = _split_scaling_params(scaling_params)
    mean_t = torch.as_tensor(means, dtype=y_norm.dtype, device=device)
    std_t = torch.as_tensor(stds, dtype=y_norm.dtype, device=device)
    return y_norm * std_t + mean_t


def _extend_h_work_after_one_step(
    H: dict,
    a_sim: torch.Tensor,
    y_norm: torch.Tensor,
    scaling_params,
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
    y_step = y_norm.view(B, -1)
    y_ch = y_step.unsqueeze(1)
    y_uns = _unscale_outputs_torch(y_step, scaling_params, device)

    active = H.get("active_entries")
    last_curr = gather_last_valid(H["current_treatments"], active).unsqueeze(1).clone()
    last_out = gather_last_valid(H["outputs"], active).unsqueeze(1).clone()

    H["prev_treatments"] = torch.cat([H["prev_treatments"], last_curr], dim=1)
    H["current_treatments"] = torch.cat([H["current_treatments"], a_sim.unsqueeze(1)], dim=1)
    H["outputs"] = torch.cat([H["outputs"], y_ch], dim=1)
    H["prev_outputs"] = torch.cat([H["prev_outputs"], last_out], dim=1)
    ae = H["active_entries"]
    H["active_entries"] = torch.cat(
        [ae, torch.ones(B, 1, ae.size(-1), device=device, dtype=ae.dtype)], dim=1
    )

    if "sequence_lengths" in H:
        H["sequence_lengths"] = H["sequence_lengths"] + 1

    if "cancer_volume" in H:
        H["cancer_volume"] = torch.cat([H["cancer_volume"], y_step[:, 0:1]], dim=1)

    if "unscaled_outputs" in H:
        uo = H["unscaled_outputs"]
        y_u = y_uns.unsqueeze(1) if uo.dim() == 3 else y_uns
        H["unscaled_outputs"] = torch.cat([uo, y_u], dim=1)

    if "chemo_application" in H:
        H["chemo_application"] = torch.cat([H["chemo_application"], a_sim[:, 0:1]], dim=1)
    if "radio_application" in H:
        H["radio_application"] = torch.cat([H["radio_application"], a_sim[:, 1:2]], dim=1)

    if "static_features" in H:
        sf = H["static_features"]
        if sf.dim() == 3:
            last = gather_last_valid(sf, active).unsqueeze(1)
            H["static_features"] = torch.cat([sf, last], dim=1)

    if "vitals" in H and "future_vitals" in H and H["future_vitals"].size(1) > 0:
        next_vitals = H["future_vitals"][:, :1, :]
        H["vitals"] = torch.cat([H["vitals"], next_vitals], dim=1)
        H["future_vitals"] = H["future_vitals"][:, 1:, :]
        if "current_covariates" in H:
            H["current_covariates"] = torch.cat([H["current_covariates"], next_vitals], dim=1)
        if "next_covariates" in H:
            H["next_covariates"] = H["next_covariates"][:, 1:, :]
        if "next_vitals" in H:
            H["next_vitals"] = H["next_vitals"][:, 1:, :]
        return

    if "current_covariates" in H:
        cc = H["current_covariates"]
        # Keep covariates separate from the outcome channel. The outcome is
        # already present in H["outputs"]/H["prev_outputs"].
        ext = gather_last_valid(cc, active).unsqueeze(1).clone()
        H["current_covariates"] = torch.cat([cc, ext], dim=1)


def _resolve_eval_tau_list(args: DictConfig) -> List[int]:
    raw = stable_select(args, "exp.iql_eval_tau_list")
    if raw is not None:
        taus = [int(t) for t in list(raw)]
        if taus:
            return taus
    return [int(args.exp.tau)]


def _resolve_iql_ckpt(args: DictConfig, original_cwd: Path) -> Path:
    explicit = stable_select(args, "exp.iql_eval_ckpt")
    if explicit:
        p = Path(str(explicit))
        if not p.is_absolute():
            p = original_cwd / p
        return p
    seed = int(args.exp.seed)
    coeff = OmegaConf.select(args, "dataset.coeff", default=None)
    if coeff is not None:
        return original_cwd / "iql_models" / f"seed_{seed}" / f"gamma_{int(coeff)}" / "iql_planner.pt"
    name = str(OmegaConf.select(args, "dataset.name", default="dataset")).replace("/", "_")
    return original_cwd / "iql_models" / f"seed_{seed}" / name / "iql_planner.pt"


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    if bool(OmegaConf.select(args, "exp.log_config", default=False)):
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

    scaling_params = dataset_collection.train_scaling_params
    if isinstance(scaling_params, (tuple, list)) and len(scaling_params) >= 2:
        mean_ser, std_ser = scaling_params[0], scaling_params[1]
    else:
        mean_ser, std_ser = None, None
    is_mimic = "mimic" in str(args.dataset.name).lower()
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

    inference_model = InferenceModel(args).to(device)
    em_eval_ckpt = str(stable_select(args, "exp.em_eval_ckpt")).strip()
    planner_path = _resolve_iql_ckpt(args, original_cwd)
    em_path = Path(em_eval_ckpt) if em_eval_ckpt else planner_path
    if em_eval_ckpt and not em_path.is_absolute():
        em_path = original_cwd / em_path

    use_em = False
    if em_path.is_file():
        import torch as _torch
        _probe = _torch.load(str(em_path), map_location="cpu")
        use_em = is_em_checkpoint(_probe)

    if use_em:
        if not em_path.exists():
            raise FileNotFoundError(f"EM checkpoint not found: {em_path}")
        logger.info("Loading combined EM checkpoint from %s", em_path)
        planner = load_em_for_eval(inference_model, str(em_path), device)
        inference_model.eval()
    else:
        iql_ckpt = str(OmegaConf.select(args, "exp.iql_inference_ckpt", default=""))
        load_inference_checkpoint(inference_model, iql_ckpt, device)
        inference_model.eval()
        if not planner_path.exists():
            raise FileNotFoundError(
                f"IQL checkpoint not found: {planner_path}. Set exp.iql_eval_ckpt / exp.em_eval_ckpt or train first."
            )
        planner = IQLPlanner.from_checkpoint(str(planner_path), device=device)
    max_action = float(planner.cfg.max_action)
    max_tau = float(stable_select(args, "exp.max_tau"))
    if max_tau <= 0:
        raise ValueError("exp.max_tau must be positive for horizon-aware IQL evaluation.")
    autoregressive_eval = bool(stable_select(args, "exp.iql_eval_autoregressive"))
    action_eval_scale = float(stable_select(args, "exp.iql_eval_action_scale"))
    action_eval_shift = float(stable_select(args, "exp.iql_eval_action_shift"))
    action_selector = str(stable_select(args, "exp.iql_eval_action_selector"))
    action_candidate_actions = int(stable_select(args, "exp.iql_eval_candidate_actions"))
    action_q_bc_penalty = float(stable_select(args, "exp.iql_eval_q_bc_penalty"))
    action_candidate_noise_std = float(stable_select(args, "exp.iql_eval_candidate_noise_std"))
    if action_eval_scale != 1.0 or action_eval_shift != 0.0:
        logger.info(
            "IQL eval action calibration enabled: a_sim <- clip(a_sim * %.6f + %.6f, 0, 1)",
            action_eval_scale,
            action_eval_shift,
        )
    if action_selector.strip().lower() not in ("", "mean", "actor_mean"):
        logger.info(
            "IQL eval action selector enabled: selector=%s candidates=%d q_bc_penalty=%.6f noise_std=%.6f",
            action_selector,
            action_candidate_actions,
            action_q_bc_penalty,
            action_candidate_noise_std,
        )
    tau_list = _resolve_eval_tau_list(args)
    original_exp_tau = int(OmegaConf.select(args, "exp.tau", default=max(tau_list)))
    base_seed = int(args.exp.seed)
    logger.info(
        "IQL eval tau RNG protocol: gift_aligned_same_seed_per_tau=True sample_seed=%d repeats=%s",
        base_seed,
        planning_repeats(args),
    )

    mlf = VCIPMlflowTracker.from_hydra(args, stage="eval")
    mlf.tags["eval_split"] = split_name
    mlf.tags["eval_tau_list"] = ",".join(str(t) for t in tau_list)
    mlf.start(args)

    per_tau_metrics: Dict[int, Dict[str, float]] = {}
    try:
        for tau in tau_list:
            tau_seed = base_seed
            logger.info(
                f"IQL eval unified closed-loop rollout: {autoregressive_eval} "
                f"(tau={tau}, target_horizon={tau}, max_tau={max_tau}, split={split_name}, sample_seed={tau_seed})"
            )
            metrics = aggregate_iql_planner_metrics(
                planner,
                inference_model,
                dataset_collection,
                fold,
                args,
                device=device,
                tau=int(tau),
                max_tau=max_tau,
                autoregressive_eval=autoregressive_eval,
                val_batch_size=batch_size,
                log_batches=True,
                include_factual_traj_rmse=True,
                sample_seed=tau_seed,
            )
            logger.info("--- Aggregate online policy metric ---")
            logger.info(f"Split: {split_name}")
            logger.info(f"Global RMSE on stacked batches (normalized space): {float(metrics['rmse_norm']):.6f}")
            logger.info(
                f"MAE normalized: {float(metrics['mae_norm']):.6f} | "
                f"MAE unscaled: {float(metrics['mae_uns']):.6f} | "
                f"RMSE unscaled: {float(metrics['rmse_uns']):.6f}"
            )
            per_tau_metrics[int(tau)] = {
                "mae_norm": float(metrics["mae_norm"]),
                "mae_uns": float(metrics["mae_uns"]),
                "rmse_uns": float(metrics["rmse_uns"]),
                "rmse_norm": float(metrics["rmse_norm"]),
                "rmse_factual_norm": float(metrics["rmse_factual_norm"])
                if metrics.get("rmse_factual_norm") is not None
                else float("nan"),
                "mean_batch_rmse_iql": float(metrics["mean_batch_rmse_plan"])
                if metrics.get("mean_batch_rmse_plan") is not None
                else float("nan"),
                "mean_batch_rmse_factual": float(metrics["mean_batch_rmse_factual"])
                if metrics.get("mean_batch_rmse_factual") is not None
                else float("nan"),
            }

        mlf.log_eval_tau_metrics(per_tau_metrics, step=0)
    finally:
        args.exp.tau = original_exp_tau
        mlf.finish(final_step=0)


if __name__ == "__main__":
    main()
