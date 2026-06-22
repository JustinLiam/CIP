"""
Save/load CT+IQL EM checkpoints (encoder + WeightNet + IQL planner).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.models.ct_encoder_weight import CTEncoderWeightModel
from src.planners.iql_planner import IQLPlanner, IQLPlannerConfig

logger = logging.getLogger(__name__)

EM_CKPT_FORMAT = "ct_iql_em_v1"


def save_em_checkpoint(
    path: Path,
    *,
    ct_model: CTEncoderWeightModel,
    planner: IQLPlanner,
    outer_iter: int,
    config_yaml: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": EM_CKPT_FORMAT,
        "ct_history_encoder": ct_model.ct_encoder.state_dict(),
        "projection_head": ct_model.projection.state_dict(),
        "weight_net": ct_model.weight_net.state_dict(),
        "iql": planner.state_dict(),
        "outer_iter": int(outer_iter),
        "config": config_yaml,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
    logger.info("Saved EM checkpoint to %s (outer_iter=%d).", path, outer_iter)


def is_em_checkpoint(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") == EM_CKPT_FORMAT


def load_encoder_into_inference(inference_model, em_obj: Dict[str, Any]) -> None:
    """Map EM encoder weights into ``InferenceModel`` for eval compatibility."""
    inference_model.ct_history_encoder.load_state_dict(
        em_obj["ct_history_encoder"], strict=True
    )
    inference_model.projection_head.load_state_dict(
        em_obj["projection_head"], strict=True
    )
    inference_model._outcome_predictor_loaded = False
    logger.info("Loaded EM encoder into InferenceModel (no outcome_predictor).")


def load_em_planner(em_obj: Dict[str, Any], device: str) -> IQLPlanner:
    iql_state = em_obj["iql"]
    cfg_dict = dict(iql_state["cfg"])
    cfg_dict["device"] = device
    cfg_dict.setdefault("max_grad_norm", None)
    cfg_dict.setdefault("encoder_max_grad_norm", 1.0)
    cfg_dict.setdefault("adv_max", 100.0)
    cfg_dict.setdefault("weight_max", 10.0)
    cfg_dict.setdefault("actor_update", "awr")
    cfg_dict.setdefault("td3bc_q_alpha", 2.5)
    cfg_dict.setdefault("td3bc_bc_alpha", 1.0)
    cfg_dict.setdefault("goal_adapter_enabled", False)
    cfg_dict.setdefault("z_dim", None)
    cfg_dict.setdefault("output_dim", None)
    cfg_dict.setdefault("goal_adapter_hidden_dim", 64)
    cfg_dict.setdefault("goal_adapter_init_scale", 1e-3)
    planner = IQLPlanner(IQLPlannerConfig(**cfg_dict))
    planner.load_eval_weights(iql_state)
    planner.actor.eval()
    return planner


def load_em_for_eval(
    inference_model,
    ckpt_path: str,
    device: str,
) -> IQLPlanner:
    """Load combined EM checkpoint for evaluation."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"EM checkpoint not found: {ckpt_path}")
    obj = torch.load(str(path), map_location=device)
    if not is_em_checkpoint(obj):
        raise ValueError(f"Not an EM checkpoint (format={obj.get('format')!r}): {ckpt_path}")
    load_encoder_into_inference(inference_model, obj)
    return load_em_planner(obj, device)


def load_em_ct_model(
    ct_model: CTEncoderWeightModel,
    ckpt_path: str,
    device: str,
) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device)
    if not is_em_checkpoint(obj):
        raise ValueError(f"Not an EM checkpoint: {ckpt_path}")
    ct_model.load_state_dict_encoder(
        {
            "ct_history_encoder": obj["ct_history_encoder"],
            "projection_head": obj["projection_head"],
            "weight_net": obj["weight_net"],
        }
    )
    return obj
