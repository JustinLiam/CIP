"""
Optional MLflow tracking for VCIP pipelines (CT / IQL / eval).

When ``exp.use_mlflow`` is false (default), all methods are no-ops and mlflow is not imported.
Failures to reach the tracking server are logged and do not interrupt training.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class VCIPMlflowTracker:
    """Thin MLflow wrapper; safe to instantiate with ``enabled=False``."""

    def __init__(
        self,
        *,
        enabled: bool,
        experiment_name: str = "vcip_tumor",
        tracking_uri: Optional[str] = None,
        log_artifacts: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.experiment_name = str(experiment_name)
        self.tracking_uri = tracking_uri
        self.log_artifacts = bool(log_artifacts)
        self.tags = {k: str(v) for k, v in (tags or {}).items()}
        self._active = False
        self._run_id: Optional[str] = None
        self._metrics_logged = 0

    @classmethod
    def from_hydra(cls, args: Any, *, stage: str) -> "VCIPMlflowTracker":
        """Build tracker from Hydra ``args`` (OmegaConf DictConfig)."""
        try:
            from omegaconf import OmegaConf
        except ImportError:
            OmegaConf = None  # type: ignore

        def _sel(key: str, default: Any = None) -> Any:
            if OmegaConf is None:
                return default
            return OmegaConf.select(args, key, default=default)

        enabled = bool(_sel("exp.use_mlflow", default=False))
        experiment = str(_sel("exp.mlflow_experiment", default="vcip_tumor"))
        uri = _sel("exp.mlflow_uri", default=None)
        if uri is not None:
            uri = str(uri)
        log_artifacts = bool(_sel("exp.mlflow_log_artifacts", default=True))

        seed = _sel("exp.seed", default="")
        coeff = _sel("dataset.coeff", default="")
        dataset_name = _sel("dataset.name", default="")
        combo_id = _sel("exp.mlflow_combo_id", default="")
        if not combo_id:
            combo_id = _sel("exp.combo_id", default="")

        tags: Dict[str, str] = {
            "stage": str(stage),
            "seed": str(seed),
            "gamma": str(coeff),
            "dataset": str(dataset_name),
        }
        if combo_id:
            tags["combo_id"] = str(combo_id)

        return cls(
            enabled=enabled,
            experiment_name=experiment,
            tracking_uri=uri,
            log_artifacts=log_artifacts,
            tags=tags,
        )

    def _run_name(self) -> str:
        parts = [self.tags.get("stage", "run")]
        if self.tags.get("seed"):
            parts.append(f"seed{self.tags['seed']}")
        if self.tags.get("gamma"):
            parts.append(f"g{self.tags['gamma']}")
        if self.tags.get("combo_id"):
            parts.append(str(self.tags["combo_id"]))
        return "_".join(parts)

    def start(self, args: Any) -> None:
        """Start an MLflow run and log Hydra config as an artifact."""
        if not self.enabled:
            return
        try:
            import mlflow
            from omegaconf import OmegaConf

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            if mlflow.active_run() is not None:
                logger.warning("Ending stale MLflow run before starting a new one.")
                mlflow.end_run()
            run = mlflow.start_run(run_name=self._run_name())
            self._run_id = run.info.run_id
            mlflow.set_tags(self.tags)
            self._active = True
            try:
                self._log_config_artifact(args, OmegaConf)
            except Exception as exc:
                # Config artifact is optional; do not disable metric logging.
                logger.warning("MLflow config artifact failed (%s); metrics will still be logged.", exc)
            logger.info(
                "MLflow run started (experiment=%s, run_id=%s, run_name=%s)",
                self.experiment_name,
                self._run_id,
                self._run_name(),
            )
        except Exception as exc:
            logger.warning("MLflow start failed (%s); continuing without tracking.", exc)
            self._end_active_run_quietly()
            self.enabled = False
            self._active = False
            self._run_id = None

    def _log_config_artifact(self, args: Any, OmegaConf: Any) -> None:
        import mlflow

        yaml_text = OmegaConf.to_yaml(args, resolve=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="vcip_config_"
        ) as tmp:
            tmp.write(yaml_text)
            tmp_path = tmp.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="config")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        if not self.enabled or not self._active:
            return
        try:
            import mlflow

            clean = {
                k: float(v)
                for k, v in metrics.items()
                if v is not None and _is_finite(float(v))
            }
            if clean:
                mlflow.log_metrics(clean, step=int(step))
                self._metrics_logged += len(clean)
        except Exception as exc:
            logger.warning("MLflow log_metrics failed at step %s: %s", step, exc)

    def log_iql_training_step(
        self,
        step: int,
        logs: Mapping[str, float],
        loss_keys: Sequence[str],
        loss_buf: Mapping[str, Any],
        log_window: int,
        *,
        dw_log_keys: Sequence[str] = (),
        dw_buf: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Mirror train_iql_planner console logs (every 200 steps)."""
        if not self.enabled or not self._active:
            return
        metrics: Dict[str, float] = {}
        win = max(1, int(log_window))
        for k in loss_keys:
            if k not in logs:
                continue
            metrics[f"train/{k}"] = float(logs[k])
            buf = loss_buf.get(k)
            if buf:
                import numpy as np

                metrics[f"train/{k}_mean_{win}"] = float(np.mean(buf))
        if dw_buf is not None:
            for k in dw_log_keys:
                if k not in logs:
                    continue
                metrics[f"train/dw/{k}"] = float(logs[k])
                buf = dw_buf.get(k)
                if buf:
                    import numpy as np

                    metrics[f"train/dw/{k}_mean_{win}"] = float(np.mean(buf))
        self.log_metrics(metrics, step=step)

    def log_iql_val_step(
        self,
        step: int,
        per_world: Mapping[str, Mapping[str, float]],
        val_worlds: Sequence[str],
        val_metric_key: str,
        *,
        improved_worlds: Optional[Sequence[str]] = None,
    ) -> None:
        improved = set(improved_worlds or ())
        metrics: Dict[str, float] = {}
        for w in val_worlds:
            wm = per_world[w]
            prefix = f"val/{w}"
            metrics[f"{prefix}/mae_norm"] = float(wm["mae_norm"])
            metrics[f"{prefix}/mae_uns"] = float(wm["mae_uns"])
            metrics[f"{prefix}/rmse_norm"] = float(wm["rmse_norm"])
            metrics[f"{prefix}/{val_metric_key}"] = float(wm[val_metric_key])
            if w in improved:
                metrics[f"{prefix}/best_improved"] = 1.0
        self.log_metrics(metrics, step=step)

    def _end_active_run_quietly(self) -> None:
        try:
            import mlflow

            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:
            pass

    def finish(
        self,
        *,
        artifact_paths: Optional[Sequence[Optional[PathLike]]] = None,
        final_metrics: Optional[Mapping[str, float]] = None,
        final_step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            import mlflow

            if self._active:
                if final_metrics and final_step is not None:
                    self.log_metrics(final_metrics, step=int(final_step))
                if self.log_artifacts and artifact_paths:
                    for p in artifact_paths:
                        if p is None:
                            continue
                        path = Path(p)
                        if path.is_file():
                            try:
                                mlflow.log_artifact(str(path.resolve()))
                            except Exception as exc:
                                logger.warning("MLflow log_artifact failed for %s: %s", path, exc)
                        elif path.exists():
                            logger.warning("MLflow skip artifact (not a file): %s", path)
                if self._metrics_logged == 0:
                    logger.warning(
                        "MLflow run %s ended with ZERO metrics logged. "
                        "Check training logs for 'MLflow start failed' or 'log_metrics failed', "
                        "and confirm exp.use_mlflow=true in the process that ran training.",
                        self._run_id,
                    )
                else:
                    logger.info(
                        "MLflow run ended (run_id=%s, metrics_logged=%d).",
                        self._run_id,
                        self._metrics_logged,
                    )
                mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow finish failed: %s", exc)
            self._end_active_run_quietly()
        finally:
            self._active = False
            self._run_id = None


def _is_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")

