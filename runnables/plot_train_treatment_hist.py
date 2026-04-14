"""
Histogram + normality diagnostics for train-split treatments (active timesteps only).

Run:
  python runnables/plot_train_treatment_hist.py +dataset=cancer_sim_cont +model=vcip

Outputs:
  plots/train_treatment_active_hist.png   — histograms + dashed N(μ̂,σ̂) overlay
  plots/train_treatment_qq_normal.png    — Q–Q plot vs standard normal (of standardized data)

How to check "long tail vs normal":
  1) Q–Q plot: points should fall on the diagonal if Gaussian. S-curve → skew; tails bend up/down → heavy/light tails.
  2) Moments: skewness ≠ 0, excess kurtosis ≠ 0 → not normal.
  3) Tests (large n → almost always reject for real data): D'Agostino–Pearson, K–S vs fitted normal, Anderson–Darling.
  4) Treatments live in [0,1]: a Gaussian is a poor *model* anyway; Beta / mixture is more natural than "long-tailed normal".
"""
import logging
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import repeat_static, set_seed, to_float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("toint", lambda x: int(x), replace=True)


def _moments(x: np.ndarray):
    x = x.astype(np.float64)
    mu = x.mean()
    sd = x.std()
    if sd < 1e-12:
        return mu, sd, float("nan"), float("nan")
    skew = float(((x - mu) ** 3).mean() / (sd**3))
    kurt = float(((x - mu) ** 4).mean() / (sd**4) - 3.0)
    return mu, sd, skew, kurt


def _normal_pdf_overlay(ax, x_flat, n_bins, color="C1", label="N(μ̂,σ̂) PDF (scaled)"):
    mu, sd, _, _ = _moments(x_flat)
    if sd < 1e-12:
        return
    counts, edges = np.histogram(x_flat, bins=n_bins, density=False)
    width = edges[1] - edges[0]
    scale = (counts.sum() * width) if counts.sum() > 0 else 1.0
    xs = np.linspace(edges[0], edges[-1], 200)
    pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
    ax.plot(xs, pdf * scale, color=color, lw=2, ls="--", label=label)


def _log_normality_suite(name: str, vals: np.ndarray, seed: int):
    """Print tests; safe for large n (subsample where needed)."""
    from scipy import stats as scipy_stats

    mu, sd, skew, ek = _moments(vals)
    logger.info("[%s] skewness=%.4f excess_kurtosis=%.4f (normal: 0, 0)", name, skew, ek)

    sub = vals
    if sub.size > 5000:
        rng = np.random.default_rng(seed)
        sub = rng.choice(sub, size=5000, replace=False)
    if sub.size >= 8:
        stat, p = scipy_stats.normaltest(sub)
        logger.info("  D'Agostino-Pearson (n=%d): stat=%.4f p=%.4g", sub.size, stat, p)
    if sub.size >= 3 and sub.size <= 5000:
        stat_sw, p_sw = scipy_stats.shapiro(sub)
        logger.info("  Shapiro-Wilk (n=%d): stat=%.4f p=%.4g", sub.size, stat_sw, p_sw)
    elif sub.size > 5000:
        sub_sw = np.random.default_rng(seed).choice(vals, size=5000, replace=False)
        stat_sw, p_sw = scipy_stats.shapiro(sub_sw)
        logger.info("  Shapiro-Wilk (n=5000 subsample): stat=%.4f p=%.4g", stat_sw, p_sw)

    # K–S: empirical vs *fitted* Gaussian (μ̂,σ̂ from same data → p-values are approximate only)
    if sd > 1e-12 and vals.size > 10:
        stat_ks, p_ks = scipy_stats.kstest(vals, "norm", args=(mu, sd))
        logger.info("  K-S vs N(μ̂,σ̂) (full n=%d): stat=%.6f p=%.4g", vals.size, stat_ks, p_ks)

    if vals.size >= 15:
        ad = scipy_stats.anderson(vals, dist="norm")
        # ad.significance_level: [15%, 10%, 5%, 2.5%, 1%]; ad.critical_values aligned
        logger.info(
            "  Anderson-Darling: statistic=%.4f (critical at 5%%=%.4f)",
            float(ad.statistic),
            float(ad.critical_values[2]) if len(ad.critical_values) > 2 else float("nan"),
        )


@hydra.main(version_base=None, config_name="config.yaml", config_path="../configs/")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    seed = int(args.exp.seed)
    set_seed(seed)
    original_cwd = Path(get_original_cwd())
    args["exp"]["processed_data_dir"] = os.path.join(str(original_cwd), args["exp"]["processed_data_dir"])

    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    dataset_collection = to_float(dataset_collection)
    if args["dataset"]["static_size"] > 0:
        dims = len(dataset_collection.train_f.data["static_features"].shape)
        if dims == 2:
            dataset_collection = repeat_static(dataset_collection)

    data = dataset_collection.train_f.data
    ct = np.asarray(data["current_treatments"], dtype=np.float64)
    ae = np.asarray(data["active_entries"], dtype=np.float64)
    if ae.ndim == 3:
        mask = ae.squeeze(-1) > 0.5
    else:
        mask = ae > 0.5

    if ct.ndim != 3:
        raise ValueError(f"Expected current_treatments [N,T,A], got shape {ct.shape}")

    n_treat = ct.shape[-1]
    names = ["chemo (dim 0)", "radio (dim 1)"]
    if n_treat > 2:
        names = [f"dim {d}" for d in range(n_treat)]

    vals_per_dim = []
    for d in range(n_treat):
        v = ct[:, :, d][mask]
        v = v[np.isfinite(v)]
        vals_per_dim.append(v)
        logger.info("%s: n=%d min=%.6f max=%.6f mean=%.6f", names[d], v.size, v.min(), v.max(), v.mean())
        _log_normality_suite(names[d], v, seed)

    out_dir = original_cwd / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bins = int(OmegaConf.select(args, "exp.treatment_hist_bins", default=40))

    # --- Histograms ---
    fig, axes = plt.subplots(1, n_treat, figsize=(5 * n_treat, 4), squeeze=False)
    for d, vals in enumerate(vals_per_dim):
        ax = axes[0, d]
        ax.hist(vals, bins=n_bins, density=False, color="steelblue", edgecolor="white", alpha=0.85)
        _normal_pdf_overlay(ax, vals, n_bins)
        ax.set_title(names[d])
        ax.set_xlabel("treatment value")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Train: active timesteps — current_treatments", fontsize=11)
    fig.tight_layout()
    p_hist = out_dir / "train_treatment_active_hist.png"
    fig.savefig(p_hist, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p_hist)

    # --- Q–Q vs normal (standardize each sample for comparable axes) ---
    fig2, axes2 = plt.subplots(1, n_treat, figsize=(5 * n_treat, 4), squeeze=False)
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        logger.warning("scipy missing; skip Q-Q plot")
        return

    for d, vals in enumerate(vals_per_dim):
        ax = axes2[0, d]
        mu, sd, _, _ = _moments(vals)
        if sd < 1e-12:
            ax.text(0.5, 0.5, "constant", ha="center")
            continue
        z = (vals - mu) / sd
        scipy_stats.probplot(z, dist="norm", plot=ax)
        ax.set_title(f"{names[d]}: Q–Q vs N(0,1)\n(standardized)")
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_alpha(0.35)
    fig2.suptitle(
        "If normal: points on line. Tail bends → heavy/light tails; S-shape → skew.",
        fontsize=10,
    )
    fig2.tight_layout()
    p_qq = out_dir / "train_treatment_qq_normal.png"
    fig2.savefig(p_qq, dpi=150)
    plt.close(fig2)
    logger.info("Saved %s", p_qq)


if __name__ == "__main__":
    main()
