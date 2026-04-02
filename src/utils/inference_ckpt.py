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
