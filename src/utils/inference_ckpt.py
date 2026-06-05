"""
Load weights into ``InferenceModel`` from:
  - ``train_ct.py`` → ``ct_best_encoder.pt`` (``ct_history_encoder`` + ``projection_head`` only)
  - Full ``VAEModel.state_dict()`` with ``inference_model.*`` keys
  - Raw ``InferenceModel.state_dict()``-style flat dict
"""
import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def _infer_mlp_hidden_dim(state_dict: Dict[str, torch.Tensor], weight_key: str = "net.0.weight") -> int:
    """First Linear in CT MLP heads: out_features = hidden_dim."""
    w = state_dict[weight_key]
    return int(w.shape[0])


def _attach_latent_rollout_modules(
    inference_model,
    obj: Dict[str, Any],
) -> bool:
    """
    Build ``z_dynamics`` / ``outcome_decoder`` with widths matching the CT checkpoint, then load weights.
    Returns True on success.
    """
    from src.models.ct_deconfound import LatentDynamicsPredictor, OutcomeDecoder

    if "z_dynamics" not in obj or "outcome_decoder" not in obj:
        return False

    z_dim = int(getattr(inference_model, "z_dim", 0))
    a_dim = int(getattr(inference_model, "treatment_dim", getattr(inference_model, "treatment_size", 0)))
    y_dim = int(getattr(inference_model, "output_dim", 0))

    z_sd = obj["z_dynamics"]
    dec_sd = obj["outcome_decoder"]
    ph = int(obj.get("predictor_hidden", 64))
    dyn_hidden = int(obj.get("dyn_hidden", _infer_mlp_hidden_dim(z_sd)))
    dec_hidden = int(obj.get("decoder_hidden", obj.get("predictor_hidden", _infer_mlp_hidden_dim(dec_sd))))
    dyn_residual = bool(obj.get("dyn_residual", obj.get("ct_dyn_residual", True)))

    if hasattr(inference_model, "z_dynamics"):
        del inference_model.z_dynamics
    if hasattr(inference_model, "outcome_decoder"):
        del inference_model.outcome_decoder

    inference_model.add_module(
        "z_dynamics",
        LatentDynamicsPredictor(z_dim, a_dim, hidden_dim=dyn_hidden, residual=dyn_residual),
    )
    inference_model.add_module(
        "outcome_decoder",
        OutcomeDecoder(z_dim, y_dim, hidden_dim=dec_hidden),
    )
    inference_model.z_dynamics.load_state_dict(z_sd, strict=True)
    inference_model.outcome_decoder.load_state_dict(dec_sd, strict=True)
    inference_model._ct_rollout_loaded = True
    inference_model._ct_rollout_mode = obj.get("rollout_mode", "latent_dynamics")
    inference_model._ct_rollout_k_max = int(obj.get("rollout_k_max", 1))
    inference_model._ct_dyn_hidden = dyn_hidden
    inference_model._ct_decoder_hidden = dec_hidden
    logger.info(
        "Loaded z_dynamics (hidden=%s, residual=%s) + outcome_decoder (hidden=%s) "
        "from checkpoint (rollout_mode=%s).",
        dyn_hidden,
        dyn_residual,
        dec_hidden,
        inference_model._ct_rollout_mode,
    )
    return True


def load_inference_checkpoint(inference_model, ckpt_path: str, device: str) -> None:
    if not ckpt_path:
        logger.info("No inference checkpoint path. Using randomly initialized CT encoder + projection.")
        return
    path = Path(ckpt_path)
    if not path.exists():
        logger.warning("Checkpoint not found: %s. Using initialized weights.", ckpt_path)
        return

    obj: Any = torch.load(str(path), map_location=device)

    # train_ct.py → ct_best_encoder.pt
    if isinstance(obj, dict) and "ct_history_encoder" in obj and "projection_head" in obj:
        m_ce, u_ce = inference_model.ct_history_encoder.load_state_dict(obj["ct_history_encoder"], strict=True)
        m_ph, u_ph = inference_model.projection_head.load_state_dict(obj["projection_head"], strict=True)
        logger.info("Loaded train_ct encoder weights from %s (ct_history_encoder + projection_head).", ckpt_path)
        if m_ce or u_ce:
            logger.info("ct_history_encoder missing=%s unexpected=%s", m_ce, u_ce)
        if m_ph or u_ph:
            logger.info("projection_head missing=%s unexpected=%s", m_ph, u_ph)

        # Optional: load OutcomePredictor for downstream model-based OPE in IQL val/eval.
        if hasattr(inference_model, "outcome_predictor") and "outcome_predictor" in obj:
            try:
                m_op, u_op = inference_model.outcome_predictor.load_state_dict(
                    obj["outcome_predictor"], strict=True
                )
                inference_model._outcome_predictor_loaded = True
                logger.info("Loaded outcome_predictor weights from %s.", ckpt_path)
                if m_op or u_op:
                    logger.info("outcome_predictor missing=%s unexpected=%s", m_op, u_op)
            except Exception as e:  # shape mismatch: fall back silently but flag it
                inference_model._outcome_predictor_loaded = False
                logger.warning(
                    "outcome_predictor state_dict in %s could not be loaded (%s). "
                    "Predictor-world rollouts will use randomly initialized weights.",
                    ckpt_path, e,
                )
        elif hasattr(inference_model, "outcome_predictor"):
            inference_model._outcome_predictor_loaded = False
            logger.warning(
                "Checkpoint %s does not contain 'outcome_predictor' weights. "
                "Re-train CT to enable predictor-based model-based OPE (A/B val); "
                "simulator-based val will still work.",
                ckpt_path,
            )

        # Optional latent-rollout modules (train_ct ct_rollout_mode=latent_dynamics).
        inference_model._ct_rollout_loaded = False
        if "z_dynamics" in obj and "outcome_decoder" in obj:
            try:
                _attach_latent_rollout_modules(inference_model, obj)
            except Exception as e:
                inference_model._ct_rollout_loaded = False
                logger.warning(
                    "Could not load latent rollout modules from %s (%s). "
                    "IQL will use outcome_predictor only.",
                    ckpt_path,
                    e,
                )
        elif "rollout_mode" in obj:
            inference_model._ct_rollout_mode = obj.get("rollout_mode", "none")
        return

    state_dict: Dict[str, torch.Tensor] = obj
    if any(k.startswith("inference_model.") for k in state_dict.keys()):
        inf_sd = {
            k.replace("inference_model.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("inference_model.")
        }
        missing, unexpected = inference_model.load_state_dict(inf_sd, strict=False)
    else:
        missing, unexpected = inference_model.load_state_dict(state_dict, strict=False)

    logger.info("Loaded inference checkpoint: %s", ckpt_path)
    if missing:
        logger.info("Missing keys: %s%s", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        logger.info("Unexpected keys: %s%s", unexpected[:5], "..." if len(unexpected) > 5 else "")
