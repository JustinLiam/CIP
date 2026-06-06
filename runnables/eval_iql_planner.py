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
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from torch.distributions import Distribution
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.iql_dataset_builder import align_h_t_static_to_history, dataset_actions_to_tanh_policy_space
from src.models.inference_model import InferenceModel
from src.planners.iql_planner import IQLPlanner
from src.utils.inference_ckpt import load_inference_checkpoint
from src.utils.mlflow_vcip import VCIPMlflowTracker
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _default_iql_ckpt_candidates(args: DictConfig, original_cwd: Path) -> list[Path]:
    """
    Resolve IQL eval checkpoint candidates:
    1) New dataset-aware layout (primary).
    2) Legacy gamma layout (fallback for old experiments).
    """
    seed = int(OmegaConf.select(args, "exp.seed", default=0))
    dataset_name = str(OmegaConf.select(args, "dataset.name", default="dataset"))
    candidates = [
        original_cwd / "iql_models" / dataset_name / f"seed_{seed}" / "iql_planner.pt",
    ]
    coeff = OmegaConf.select(args, "dataset.coeff", default=None)
    if coeff is not None:
        coeff_str = str(coeff)
        candidates.insert(
            0,
            original_cwd / "iql_models" / dataset_name / f"seed_{seed}" / f"coeff_{coeff_str}" / "iql_planner.pt",
        )
        # Legacy location used by existing cancer experiments.
        try:
            coeff_int = int(coeff)
            candidates.append(
                original_cwd / "iql_models" / f"seed_{seed}" / f"gamma_{coeff_int}" / "iql_planner.pt"
            )
        except (TypeError, ValueError):
            pass
    return candidates


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


def _sim_actions_to_tanh_batch(a_sim: torch.Tensor, max_action: float) -> torch.Tensor:
    """Match ``dataset_actions_to_tanh_policy_space`` for batched simulator actions [B, A]."""
    if max_action <= 0:
        return a_sim
    a = torch.clamp(a_sim, 0.0, max_action)
    return 2.0 * a - max_action


def _iql_augmented_state(
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
    return torch.cat([z, eval_target, delta, a_prev_tanh], dim=-1)


def _resolve_outcome_scaling(mean_ser, std_ser, preferred_key: str = "outputs") -> tuple[float, float, str]:
    for key in (preferred_key, "cancer_volume"):
        try:
            return float(mean_ser[key]), float(std_ser[key]), key
        except Exception:
            pass
    return 0.0, 1.0, "none"


def _unscaled_outcome(y_norm: torch.Tensor, mean_ser, std_ser) -> torch.Tensor:
    """y_norm: [B, 1] normalized outcome -> unscaled [B, 1]."""
    m, s, _ = _resolve_outcome_scaling(mean_ser, std_ser, preferred_key="outputs")
    return y_norm * s + m


def _unscaled_outcome_np(y_norm: np.ndarray, mean_ser, std_ser) -> np.ndarray:
    """Same as ``_unscaled_outcome`` for numpy (e.g. simulator outputs)."""
    m, s, _ = _resolve_outcome_scaling(mean_ser, std_ser, preferred_key="outputs")
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
    y_uns = _unscaled_outcome(y_col, mean_ser, std_ser)

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
    for candidate in _default_iql_ckpt_candidates(args, original_cwd):
        if candidate.exists():
            return candidate
    return _default_iql_ckpt_candidates(args, original_cwd)[0]


def _resolve_eval_tau_list(args: DictConfig) -> List[int]:
    """Single ``exp.tau`` unless ``exp.iql_eval_tau_list`` is set (multi-horizon, one MLflow run)."""
    raw = OmegaConf.select(args, "exp.iql_eval_tau_list", default=None)
    if raw is not None:
        taus = [int(t) for t in list(raw)]
        if taus:
            return taus
    return [int(args.exp.tau)]


def _evaluate_iql_at_tau(
    tau: int,
    *,
    dataloader,
    fold,
    device: str,
    inference_model: InferenceModel,
    planner: IQLPlanner,
    max_action: float,
    max_tau: float,
    autoregressive_eval: bool,
    mean_ser,
    std_ser,
    std_outcome: float,
    outcome_scale_key: str,
    split_name: str,
) -> Dict[str, float]:
    """Run closed-loop IQL eval at horizon ``tau``; return aggregate metrics."""
    logger.info(
        f"IQL eval autoregressive action rollout: {autoregressive_eval} (tau={tau}, max_tau={max_tau})"
    )

    losses: List[float] = []
    losses_2: List[float] = []
    ture_output_list: List[np.ndarray] = []
    output_after_actions_list: List[np.ndarray] = []
    ture_output_actions_list: List[np.ndarray] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            H_t, targets = batch
            H_t = align_h_t_static_to_history(H_t)
            for key in H_t:
                H_t[key] = H_t[key].to(device)
            for key in targets:
                targets[key] = targets[key].to(device)

            if autoregressive_eval:
                eval_target = targets["outputs"][:, -1, :]
                H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                a_prev_sim = H_work["current_treatments"][:, -1, :].clone()
                planned = []
                for step in range(tau):
                    H_work = align_h_t_static_to_history(H_work)
                    z, _, _ = inference_model.ct_hidden_history(H_work)
                    a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
                    obs = _iql_augmented_state(z, eval_target, step, tau, max_tau, a_prev_tanh)
                    po = planner.actor(obs)
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
                        (mean_ser, std_ser),
                    )
                    y_norm = torch.as_tensor(y_np, device=device, dtype=torch.float32)
                    _extend_h_work_after_one_step(
                        H_work, a_sim, y_norm, mean_ser, std_ser, torch.device(device)
                    )
                    a_prev_sim = a_sim
                a_seq = torch.stack(planned, dim=1).contiguous()
            else:
                z, _, _ = inference_model.ct_hidden_history(H_t)
                z_np = z.detach().cpu().numpy()
                eval_target_np = targets["outputs"][:, -1, :].detach().cpu().numpy()
                a_prev_raw = H_t["current_treatments"][:, -1, :].detach().cpu().numpy()
                a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
                bsz = z_np.shape[0]
                delta_scalar = float(tau - 0) / max_tau
                delta_vec = np.array([delta_scalar], dtype=np.float32)
                a_rows = []
                for b in range(bsz):
                    obs_b = np.concatenate([z_np[b], eval_target_np[b], delta_vec, a_prev_feat[b]], axis=0)
                    a_rows.append(planner.act(obs_b))
                a_raw = np.stack(a_rows, axis=0)
                a_sim = _actions_to_sim_interval(a_raw, max_action)
                a_seq = torch.tensor(a_sim, device=device, dtype=torch.float32).unsqueeze(1).expand(-1, tau, -1).contiguous()

            output_after_actions = fold.simulate_output_after_actions(
                H_t, a_seq, (mean_ser, std_ser)
            )
            ture_output = targets["outputs"][:, -1, :].detach().cpu().numpy()
            loss = np.sqrt(((output_after_actions - ture_output) ** 2).mean())
            losses.append(loss)

            true_actions = targets["current_treatments"]
            ture_output_actions = fold.simulate_output_after_actions(
                H_t, true_actions, (mean_ser, std_ser)
            )
            loss_2 = np.sqrt(((ture_output_actions - ture_output) ** 2).mean())
            losses_2.append(loss_2)

            ture_output_list.append(ture_output)
            output_after_actions_list.append(output_after_actions)
            ture_output_actions_list.append(ture_output_actions)

            logger.info(f"Batch {i} RMSE (IQL plan): {loss:.6f}, RMSE (factual actions): {loss_2:.6f}")

    ture_output_arr = np.concatenate(ture_output_list, axis=0)
    output_after_actions_arr = np.concatenate(output_after_actions_list, axis=0)
    ture_output_actions_arr = np.concatenate(ture_output_actions_list, axis=0)

    rmse_norm = float(np.sqrt(((output_after_actions_arr - ture_output_arr) ** 2).mean()))
    rmse_factual_norm = float(np.sqrt(((ture_output_actions_arr - ture_output_arr) ** 2).mean()))

    iql_y_norm = output_after_actions_arr.reshape(-1)
    true_y_norm = ture_output_arr.reshape(-1)
    iql_y_uns = _unscaled_outcome_np(output_after_actions_arr, mean_ser, std_ser).reshape(-1)
    true_y_uns = _unscaled_outcome_np(ture_output_arr, mean_ser, std_ser).reshape(-1)

    mae_norm = float(np.mean(np.abs(iql_y_norm - true_y_norm)))
    mae_unscaled = float(np.mean(np.abs(iql_y_uns - true_y_uns)))
    rmse_unscaled = float(np.sqrt(np.mean((iql_y_uns - true_y_uns) ** 2)))

    logger.info("--- Aggregate (same protocol as optimize_interventions_onetime) ---")
    logger.info(f"Split: {split_name}")
    logger.info(f"Mean per-batch RMSE (IQL): {float(np.mean(losses)):.6f}")
    logger.info(f"Mean per-batch RMSE (factual traj): {float(np.mean(losses_2)):.6f}")
    logger.info(f"Global RMSE on stacked batches (normalized space): {rmse_norm:.6f}")
    logger.info(f"Global RMSE × std (outcome scale, VCIP-style): {rmse_norm * std_outcome:.6f}")
    logger.info(f"Factual global RMSE × std: {rmse_factual_norm * std_outcome:.6f}")

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
        f"Train scaling outcome key={outcome_scale_key}: mean={_resolve_outcome_scaling(mean_ser, std_ser)[0]:.6f} "
        f"std={_resolve_outcome_scaling(mean_ser, std_ser)[1]:.6f}"
    )
    logger.info(
        f"MAE normalized: {mae_norm:.6f} | MAE unscaled: {mae_unscaled:.6f} | RMSE unscaled: {rmse_unscaled:.6f}"
    )

    return {
        "mae_norm": mae_norm,
        "mae_uns": mae_unscaled,
        "rmse_uns": rmse_unscaled,
        "rmse_norm": rmse_norm,
        "rmse_factual_norm": rmse_factual_norm,
        "mean_batch_rmse_iql": float(np.mean(losses)),
        "mean_batch_rmse_factual": float(np.mean(losses_2)),
    }


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

    _, std, outcome_scale_key = _resolve_outcome_scaling(
        dataset_collection.train_scaling_params[0],
        dataset_collection.train_scaling_params[1],
        preferred_key="outputs",
    )

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
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    if max_tau <= 0:
        raise ValueError("exp.max_tau must be positive for horizon-aware IQL evaluation.")
    autoregressive_eval = bool(OmegaConf.select(args, "exp.iql_eval_autoregressive", default=True))
    mean_ser, std_ser = dataset_collection.train_scaling_params
    tau_list = _resolve_eval_tau_list(args)
    logger.info(f"IQL eval horizons: {tau_list} (split={split_name})")

    mlf = VCIPMlflowTracker.from_hydra(args, stage="eval")
    mlf.tags["eval_split"] = split_name
    mlf.tags["eval_tau_list"] = ",".join(str(t) for t in tau_list)
    mlf.start(args)

    per_tau_metrics: Dict[int, Dict[str, float]] = {}
    try:
        for tau in tau_list:
            logger.info(f"========== eval tau={tau} split={split_name} ==========")
            per_tau_metrics[tau] = _evaluate_iql_at_tau(
                tau,
                dataloader=dataloader,
                fold=fold,
                device=device,
                inference_model=inference_model,
                planner=planner,
                max_action=max_action,
                max_tau=max_tau,
                autoregressive_eval=autoregressive_eval,
                mean_ser=mean_ser,
                std_ser=std_ser,
                std_outcome=std,
                outcome_scale_key=outcome_scale_key,
                split_name=split_name,
            )
        mlf.log_eval_tau_metrics(per_tau_metrics, step=0)
    finally:
        mlf.finish(final_step=0)


if __name__ == "__main__":
    main()
