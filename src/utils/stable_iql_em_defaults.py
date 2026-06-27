"""Fixed one-stage CT+IQL EM defaults.

The public Hydra model config exposes only method structure and routine run
settings. Hyperparameters that are no longer swept live here as code defaults,
so they are maintained in one place without cluttering the YAML file.
"""
from typing import Any, Dict

from omegaconf import OmegaConf


STABLE_IQL_EM_DEFAULTS: Dict[str, Any] = {
    # One-stage local/global encoder structure
    "model.z_dim": 16,
    "model.inference.hidden_dim": 16,
    "model.inference.num_layers": 2,
    "model.inference.do": True,
    "model.inference.local_conv_layers": 1,
    "model.inference.local_conv_kernel_size": 6,
    "model.inference.local_conv_dilation": 1,
    # IQL replay and reward construction
    "exp.iql_batch_size": 256,
    "exp.iql_max_patients": None,
    "exp.iql_reward_type": "negative_outcome",
    "exp.iql_reward_huber_delta": 1.0,
    "exp.iql_reward_clip": 3.0,
    "exp.iql_reward_scale": "auto",
    "exp.iql_max_action": 1.0,
    "exp.iql_dataset_actions_unit_interval": True,
    "exp.iql_hidden_dim": 256,
    "exp.iql_n_hidden": 2,
    "exp.iql_tau": 0.7,
    "exp.iql_beta": 2.0,
    "exp.iql_adv_max": 100.0,
    "exp.iql_weight_max": 3.0,
    "exp.iql_actor_update": "awr",
    "exp.iql_actor_bc_loss": "expectile",
    "exp.iql_actor_bc_expectile": 0.8,
    "exp.iql_td3bc_q_alpha": 0.0,
    "exp.iql_td3bc_bc_alpha": 1.0,
    "exp.iql_td3bc_action_penalty_alpha": 0.0,
    "exp.iql_cql_alpha": 0.0,
    "exp.iql_cql_n_actions": 10,
    "exp.iql_q_high_action_penalty_alpha": 0.0,
    "exp.iql_q_high_action_penalty_margin": 0.0,
    "exp.iql_q_high_action_penalty_n_actions": 1,
    "exp.iql_discount": 0.99,
    "exp.iql_target_tau": 0.005,
    "exp.iql_actor_lr": 3.0e-4,
    "exp.iql_qf_lr": 3.0e-4,
    "exp.iql_vf_lr": 3.0e-4,
    "exp.iql_max_grad_norm": 5.0,
    "exp.iql_deterministic": False,
    "exp.iql_actor_dropout": None,
    "exp.iql_goal_adapter_enabled": False,
    "exp.iql_goal_adapter_hidden_dim": 64,
    "exp.iql_goal_adapter_init_scale": 1.0e-3,
    "exp.max_tau": 12.0,
    "exp.iql_target_sampling": "horizon_aligned",
    "exp.iql_target_horizons": [1, 2, 3, 4, 5, 6],
    "exp.iql_horizon_terminal_done": True,
    # One-stage EM loop
    "exp.em_outer_iters": 20,
    "exp.em_m_steps_per_outer": 1000,
    "exp.em_encoder_lr": 5.0e-5,
    "exp.em_encoder_max_grad_norm": 1.0,
    "exp.em_val_every": 1,
    "exp.em_val_metric": "rmse_uns",
    "exp.em_val_tau_list": [1, 2, 3, 4, 5, 6],
    "exp.em_val_tau_agg": "max",
    "exp.em_val_repeats": 3,
    "exp.em_val_seed_offset": 10007,
    "exp.em_save_every_eval_checkpoint": False,
    "exp.em_save_every_outer_checkpoint": False,
    "exp.em_warmup_outer_iters": 2,
    "exp.em_e_epochs": 5,
    "exp.em_e_w_lr": 0.01,
    "exp.em_e_refresh_every": 1,
    "exp.em_her_refresh_every": 0,
    "exp.em_her_samples_per_transition": 1,
    "exp.em_log_m_every": 50,
    "exp.em_encoder_diagnostics": False,
    "exp.em_encoder_diagnostics_every": 50,
    "exp.em_ckpt_dir": None,
    "exp.em_eval_ckpt": "",
    "exp.em_val_worlds": ["sim"],
    "exp.em_val_selection_world": "sim",
    # E-step WeightNet and encoder alignment
    "exp.ct_align_loss": "sinkhorn",
    "exp.ct_sinkhorn_blur": 0.01,
    "exp.ct_weight_hidden": 64,
    "exp.ct_weight_decay": 1.0e-5,
    "exp.ct_batch_size": 512,
    "exp.ct_num_workers": 0,
    "exp.ct_w_lr": 1.0e-2,
    "exp.ct_w_clip": 5.0,
    # Validation and final evaluation
    "exp.iql_val_batch_size": None,
    "exp.iql_eval_ckpt": "",
    "exp.iql_eval_tau_list": [1, 2, 3, 4, 5, 6],
    "exp.iql_eval_autoregressive": True,
    "exp.iql_eval_action_scale": 1.0,
    "exp.iql_eval_action_shift": 0.0,
    "exp.iql_eval_action_selector": "mean",
    "exp.iql_eval_candidate_actions": 16,
    "exp.iql_eval_q_bc_penalty": 0.0,
    "exp.iql_eval_candidate_noise_std": 0.25,
    "exp.iql_val_action_diagnostics": False,
    "exp.iql_val_action_grid_points": 11,
    "exp.iql_val_action_diag_max_batches": 2,
}

_MISSING = object()


def stable_default(path: str, fallback: Any = _MISSING) -> Any:
    if path in STABLE_IQL_EM_DEFAULTS:
        return STABLE_IQL_EM_DEFAULTS[path]
    if fallback is not _MISSING:
        return fallback
    return None


def stable_select(config: Any, path: str, fallback: Any = _MISSING) -> Any:
    """OmegaConf.select with the stable one-stage default for ``path``."""
    return OmegaConf.select(config, path, default=stable_default(path, fallback))
