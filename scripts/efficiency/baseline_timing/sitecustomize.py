"""Optional, synchronized inference timing for the GIFT baseline repository."""

import functools
import json
import os
import random
import sys
import threading
import time

import numpy as np


_OUTPUT = os.environ.get("EFFICIENCY_TIMING_JSONL")
_LOCK = threading.Lock()


def _synchronize():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _root_modules(model):
    if hasattr(model, "encoder") and hasattr(model, "actor"):
        return (model.encoder, model.actor)
    if hasattr(model, "modules"):
        return (model,)
    return ()


def _warmup_and_trace(original, model, call_args, call_kwargs):
    """Warm the inference path and count only parameters in executed modules."""
    import torch

    roots = _root_modules(model)
    touched = {}
    handles = []
    modes = {}

    def mark(module, _inputs):
        for parameter in module.parameters(recurse=False):
            touched[parameter.data_ptr()] = parameter.numel()

    for root in roots:
        for module in root.modules():
            modes[module] = module.training
            handles.append(module.register_forward_pre_hook(mark))

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        warmup_result = original(model, *call_args, **call_kwargs)
        _synchronize()
        del warmup_result
    finally:
        for handle in handles:
            handle.remove()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for module, training in modes.items():
            module.training = training
    return int(sum(touched.values())) if touched else None


def _wrap(cls, method_name):
    original = getattr(cls, method_name)
    if getattr(original, "_efficiency_timed", False):
        return

    @functools.wraps(original)
    def timed(self, history_dict_batch, goal_batch, dataset_collection,
              future_dict_batch, future_length=None, *args, **kwargs):
        tau = future_length
        if tau is None:
            tau = getattr(self, "future_length", getattr(self, "tau", None))
        batch_size = len(history_dict_batch)
        call_args = (
            history_dict_batch,
            goal_batch,
            dataset_collection,
            future_dict_batch,
            future_length,
            *args,
        )
        if not hasattr(self, "_efficiency_params_deploy"):
            self._efficiency_params_deploy = _warmup_and_trace(
                original, self, call_args, kwargs
            )
        _synchronize()
        started = time.perf_counter_ns()
        result = original(self, *call_args, **kwargs)
        _synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        record = {
            "method": f"{cls.__module__}.{cls.__name__}.{method_name}",
            "tau": int(tau),
            "batch_size": int(batch_size),
            "elapsed_ms": elapsed_ms,
            "episode_ms": elapsed_ms / max(batch_size, 1),
            "decision_ms": elapsed_ms / max(batch_size * int(tau), 1),
            "params_deploy": self._efficiency_params_deploy,
        }
        with _LOCK:
            with open(_OUTPUT, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return result

    timed._efficiency_timed = True
    setattr(cls, method_name, timed)


if _OUTPUT:
    # During Python's automatic sitecustomize import, the script directory may
    # not yet expose the repository's ``src`` package. Add the launch cwd
    # explicitly before importing the model classes.
    _gift_root = os.environ.get("EFFICIENCY_GIFT_ROOT", os.getcwd())
    if _gift_root not in sys.path:
        sys.path.insert(0, _gift_root)

    try:
        from src.baselines.base_model import BaseCausalModel

        _wrap(BaseCausalModel, "generate_treatment_plan_batch")
    except Exception:
        pass

    try:
        from src.gift.agents.sac_agent import SAC_HER_Agent

        _wrap(SAC_HER_Agent, "generate_treatment_plan_batch")
    except Exception:
        pass

    try:
        from src.gift.agents.scrl_agent import SCRL_Agent

        _wrap(SCRL_Agent, "generate_treatment_plan_batch")
    except Exception:
        pass
