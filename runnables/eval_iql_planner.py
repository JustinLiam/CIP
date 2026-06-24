"""
Evaluate IQL planner with the same simulator and RMSE metric as VAEModel.optimize_interventions_onetime:
  closed_loop_y = final y_norm observed during autoregressive rollout
  RMSE on normalized cancer_volume vs targets['outputs'][:, -1, :], then × std (cancer).

By default (``exp.iql_eval_autoregressive=true``) the planned sequence is built autoregressively: each step
re-encodes with ``ct_hidden_history`` after appending the chosen action and one-step simulated outcome.
The main RMSE uses the final one-step outcome from that closed-loop rollout. The legacy full-sequence replay
metric is also logged as ``sequence_replay_rmse`` for diagnostics.

Logs aggregate **t+tau** tumor volume under the IQL plan vs ``targets['outputs'][:, -1]``, in normalized
(train scaling) and unscaled (raw simulator scale) space.
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

from src.data.cip_dataset import CIPDataset, get_dataloader
from src.data.iql_dataset_builder import align_h_t_static_to_history, dataset_actions_to_tanh_policy_space
from src.evaluation.iql_action_selection import select_iql_policy_action
from src.models.inference_model import InferenceModel
from src.models.sequence_utils import gather_last_valid
from src.planners.iql_planner import IQLPlanner
from src.utils.em_ckpt import is_em_checkpoint, load_em_for_eval
from src.utils.inference_ckpt import load_inference_checkpoint
from src.utils.mlflow_vcip import VCIPMlflowTracker
from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)

GIFT_TUMOR_VOLUME_NORMALIZER = 1150.0


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


def _resolve_eval_tau_list(args: DictConfig) -> List[int]:
    raw = OmegaConf.select(args, "exp.iql_eval_tau_list", default=None)
    if raw is not None:
        taus = [int(t) for t in list(raw)]
        if taus:
            return taus
    return [int(args.exp.tau)]


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
    em_eval_ckpt = str(OmegaConf.select(args, "exp.em_eval_ckpt", default="")).strip()
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
    max_tau = float(OmegaConf.select(args, "exp.max_tau", default=12.0))
    if max_tau <= 0:
        raise ValueError("exp.max_tau must be positive for horizon-aware IQL evaluation.")
    autoregressive_eval = bool(OmegaConf.select(args, "exp.iql_eval_autoregressive", default=True))
    action_eval_scale = float(OmegaConf.select(args, "exp.iql_eval_action_scale", default=1.0))
    action_eval_shift = float(OmegaConf.select(args, "exp.iql_eval_action_shift", default=0.0))
    action_selector = str(OmegaConf.select(args, "exp.iql_eval_action_selector", default="mean"))
    action_candidate_actions = int(OmegaConf.select(args, "exp.iql_eval_candidate_actions", default=16))
    action_q_bc_penalty = float(OmegaConf.select(args, "exp.iql_eval_q_bc_penalty", default=0.0))
    action_candidate_noise_std = float(
        OmegaConf.select(args, "exp.iql_eval_candidate_noise_std", default=0.25)
    )
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
    mean_ser, std_ser = dataset_collection.train_scaling_params
    tau_list = _resolve_eval_tau_list(args)
    original_exp_tau = int(OmegaConf.select(args, "exp.tau", default=max(tau_list)))

    mlf = VCIPMlflowTracker.from_hydra(args, stage="eval")
    mlf.tags["eval_split"] = split_name
    mlf.tags["eval_tau_list"] = ",".join(str(t) for t in tau_list)
    mlf.start(args)

    per_tau_metrics: Dict[int, Dict[str, float]] = {}
    try:
        for tau in tau_list:
            args.exp.tau = int(tau)
            try:
                dataloader = get_dataloader(CIPDataset(data, args, train=False), batch_size=batch_size, shuffle=False)
            finally:
                args.exp.tau = original_exp_tau
            logger.info(
                f"IQL eval autoregressive action rollout: {autoregressive_eval} "
                f"(tau={tau}, target_horizon={tau}, max_tau={max_tau}, split={split_name})"
            )

            losses = []
            sequence_replay_losses = []
            losses_2 = []
            ture_output_list = []
            output_after_actions_list = []
            sequence_replay_output_after_actions_list = []
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
                        eval_target = targets["outputs"][:, -1, :]
                        H_work = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in H_t.items()}
                        a_prev_sim = gather_last_valid(
                            H_work["current_treatments"], H_work.get("active_entries")
                        ).clone()
                        planned = []
                        closed_loop_output_after_actions = None
                        for step in range(tau):
                            H_work = align_h_t_static_to_history(H_work)
                            z, _, _ = inference_model.ct_hidden_history(H_work)
                            a_prev_tanh = _sim_actions_to_tanh_batch(a_prev_sim, max_action)
                            obs = _iql_augmented_state(planner, z, eval_target, step, tau, max_tau, a_prev_tanh)
                            a_raw = select_iql_policy_action(
                                planner,
                                obs,
                                selector=action_selector,
                                candidate_actions=action_candidate_actions,
                                q_bc_penalty=action_q_bc_penalty,
                                candidate_noise_std=action_candidate_noise_std,
                            )
                            a_sim = _policy_to_sim_interval_torch(a_raw, max_action)
                            a_sim = _calibrate_sim_actions_torch(
                                a_sim, action_eval_scale, action_eval_shift
                            )
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
                            closed_loop_output_after_actions = y_norm.detach().cpu().numpy()
                            a_prev_sim = a_sim
                        a_seq = torch.stack(planned, dim=1).contiguous()
                    else:
                        z, _, _ = inference_model.ct_hidden_history(H_t)
                        z_np = z.detach().cpu().numpy()
                        eval_target_np = targets["outputs"][:, -1, :].detach().cpu().numpy()
                        a_prev_raw = gather_last_valid(
                            H_t["current_treatments"], H_t.get("active_entries")
                        ).detach().cpu().numpy()
                        a_prev_feat = dataset_actions_to_tanh_policy_space(a_prev_raw, max_action)
                        bsz = z_np.shape[0]
                        delta_scalar = float(tau - 0) / max_tau
                        delta_vec = np.array([delta_scalar], dtype=np.float32)
                        a_rows = []
                        for b in range(bsz):
                            z_b = torch.as_tensor(z_np[b:b + 1], device=device, dtype=torch.float32)
                            target_b = torch.as_tensor(eval_target_np[b:b + 1], device=device, dtype=torch.float32)
                            delta_b = torch.as_tensor(delta_vec.reshape(1, 1), device=device, dtype=torch.float32)
                            a_prev_b = torch.as_tensor(a_prev_feat[b:b + 1], device=device, dtype=torch.float32)
                            obs_b = planner.build_state(z_b, target_b, delta_b, a_prev_b)
                            a_b = select_iql_policy_action(
                                planner,
                                obs_b,
                                selector=action_selector,
                                candidate_actions=action_candidate_actions,
                                q_bc_penalty=action_q_bc_penalty,
                                candidate_noise_std=action_candidate_noise_std,
                            )
                            a_rows.append(a_b.detach().cpu().numpy().reshape(-1))
                        a_raw = np.stack(a_rows, axis=0)
                        a_sim = _actions_to_sim_interval(a_raw, max_action)
                        a_sim = _calibrate_sim_actions_np(
                            a_sim, action_eval_scale, action_eval_shift
                        )
                        a_seq = torch.tensor(a_sim, device=device, dtype=torch.float32).unsqueeze(1).expand(-1, tau, -1).contiguous()
                        closed_loop_output_after_actions = None

                    sequence_replay_output_after_actions = fold.simulate_output_after_actions(
                        H_t, a_seq, dataset_collection.train_scaling_params
                    )
                    if closed_loop_output_after_actions is None:
                        closed_loop_output_after_actions = sequence_replay_output_after_actions
                    output_after_actions = closed_loop_output_after_actions
                    ture_output = targets["outputs"][:, -1, :].detach().cpu().numpy()
                    loss = np.sqrt(((output_after_actions - ture_output) ** 2).mean())
                    sequence_replay_loss = np.sqrt(((sequence_replay_output_after_actions - ture_output) ** 2).mean())
                    losses.append(loss)
                    sequence_replay_losses.append(sequence_replay_loss)

                    true_actions = targets["current_treatments"]
                    ture_output_actions = fold.simulate_output_after_actions(
                        H_t, true_actions, dataset_collection.train_scaling_params
                    )
                    loss_2 = np.sqrt(((ture_output_actions - ture_output) ** 2).mean())
                    losses_2.append(loss_2)

                    ture_output_list.append(ture_output)
                    output_after_actions_list.append(output_after_actions)
                    sequence_replay_output_after_actions_list.append(sequence_replay_output_after_actions)
                    ture_output_actions_list.append(ture_output_actions)

                    logger.info(
                        f"Batch {i} RMSE (closed-loop plan): {loss:.6f}, "
                        f"RMSE (sequence replay): {sequence_replay_loss:.6f}, "
                        f"RMSE (factual actions): {loss_2:.6f}"
                    )

            ture_output_list = np.concatenate(ture_output_list, axis=0)
            output_after_actions_list = np.concatenate(output_after_actions_list, axis=0)
            sequence_replay_output_after_actions_list = np.concatenate(sequence_replay_output_after_actions_list, axis=0)
            ture_output_actions_list = np.concatenate(ture_output_actions_list, axis=0)

            rmse_norm = float(np.sqrt(((output_after_actions_list - ture_output_list) ** 2).mean()))
            sequence_replay_rmse_norm = float(np.sqrt(((sequence_replay_output_after_actions_list - ture_output_list) ** 2).mean()))
            rmse_factual_norm = float(np.sqrt(((ture_output_actions_list - ture_output_list) ** 2).mean()))

            iql_y_norm = output_after_actions_list.reshape(-1)
            true_y_norm = ture_output_list.reshape(-1)
            iql_y_uns = _unscaled_cancer_volume_np(output_after_actions_list, mean_ser, std_ser).reshape(-1)
            sequence_replay_y_uns = _unscaled_cancer_volume_np(
                sequence_replay_output_after_actions_list, mean_ser, std_ser
            ).reshape(-1)
            true_y_uns = _unscaled_cancer_volume_np(ture_output_list, mean_ser, std_ser).reshape(-1)

            mae_norm = float(np.mean(np.abs(iql_y_norm - true_y_norm)))
            mae_uns = float(np.mean(np.abs(iql_y_uns - true_y_uns)))
            rmse_uns = float(np.sqrt(np.mean((iql_y_uns - true_y_uns) ** 2)))
            sequence_replay_mae_norm = float(
                np.mean(np.abs(sequence_replay_output_after_actions_list.reshape(-1) - true_y_norm))
            )
            sequence_replay_mae_uns = float(np.mean(np.abs(sequence_replay_y_uns - true_y_uns)))
            sequence_replay_rmse_uns = float(np.sqrt(np.mean((sequence_replay_y_uns - true_y_uns) ** 2)))
            gift_tumor_norm_const = float(
                OmegaConf.select(args, "exp.gift_tumor_volume_normalizer", default=GIFT_TUMOR_VOLUME_NORMALIZER)
            )
            if gift_tumor_norm_const <= 0:
                raise ValueError("exp.gift_tumor_volume_normalizer must be positive.")
            gift_rmse = rmse_uns
            gift_rmse_percent = float(rmse_uns / gift_tumor_norm_const * 100.0)
            gift_mae_percent = float(mae_uns / gift_tumor_norm_const * 100.0)

            logger.info("--- Aggregate closed-loop online policy metric ---")
            logger.info(f"Split: {split_name}")
            logger.info(f"Mean per-batch RMSE (closed-loop IQL): {float(np.mean(losses)):.6f}")
            logger.info(f"Mean per-batch RMSE (sequence replay): {float(np.mean(sequence_replay_losses)):.6f}")
            logger.info(f"Mean per-batch RMSE (factual traj): {float(np.mean(losses_2)):.6f}")
            logger.info(f"Global RMSE on stacked batches (normalized space): {rmse_norm:.6f}")
            logger.info(f"Sequence replay RMSE on stacked batches (normalized space): {sequence_replay_rmse_norm:.6f}")
            logger.info(f"Global RMSE × std (cancer volume scale, VCIP-style): {rmse_norm * std:.6f}")
            logger.info(f"Sequence replay RMSE × std: {sequence_replay_rmse_norm * std:.6f}")
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
            logger.info(
                f"MAE normalized: {mae_norm:.6f} | MAE unscaled: {mae_uns:.6f} | RMSE unscaled: {rmse_uns:.6f}"
            )
            logger.info(
                f"GIFT-style tumor RMSE unscaled: {gift_rmse:.6f} | "
                f"GIFT-style tumor RMSE (% of {gift_tumor_norm_const:g}): {gift_rmse_percent:.6f} | "
                f"GIFT-style tumor MAE (% of {gift_tumor_norm_const:g}): {gift_mae_percent:.6f}"
            )

            per_tau_metrics[tau] = {
                "mae_norm": mae_norm,
                "mae_uns": mae_uns,
                "rmse_uns": rmse_uns,
                "rmse_norm": rmse_norm,
                "closed_loop_rmse": rmse_uns,
                "closed_loop_rmse_uns": rmse_uns,
                "closed_loop_rmse_norm": rmse_norm,
                "sequence_replay_rmse": sequence_replay_rmse_uns,
                "sequence_replay_rmse_uns": sequence_replay_rmse_uns,
                "sequence_replay_rmse_norm": sequence_replay_rmse_norm,
                "sequence_replay_mae_uns": sequence_replay_mae_uns,
                "sequence_replay_mae_norm": sequence_replay_mae_norm,
                "gift_rmse": gift_rmse,
                "gift_rmse_percent": gift_rmse_percent,
                "gift_mae_percent": gift_mae_percent,
                "rmse_factual_norm": rmse_factual_norm,
                "mean_batch_rmse_iql": float(np.mean(losses)),
                "mean_batch_rmse_factual": float(np.mean(losses_2)),
            }

        mlf.log_eval_tau_metrics(per_tau_metrics, step=0)
    finally:
        args.exp.tau = original_exp_tau
        mlf.finish(final_step=0)


if __name__ == "__main__":
    main()
