import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText

from scipy import stats
from scipy.stats import spearmanr, pearsonr, gaussian_kde, rankdata

warnings.filterwarnings('ignore')

from sigma_core import (
    Config, PUB_RC, C, PS_COLORS, PS_MARKERS, CH_COLORS,
    savefig, pstars, make_parser,
    discover_images, build_latent_cache,
    compute_patch_stats_mean, compute_patch_stats_per_channel,
    decompose_sigma,
)

plt.rcParams.update(PUB_RC)

N_CHANNELS  = 4
PATCH_SIZES = [8, 16, 32, 64]

# 散点图配色
SCATTER_CMAPS = {
    'mean': 'Blues',
    0: 'Oranges', 1: 'Greens', 2: 'Reds', 3: 'Purples',
}
REG_COLOR   = '#1a1a2e'
LOWESS_COLOR= '#c0392b'
CI_ALPHA    = 0.18


def extract_patch_data(latent_cache: list, patch_size: int):
    """
    提取每个 patch 的 σ（均值 + 4 个通道）和重建误差。
    返回 sigma_mean, sigma_ch, rec_all, per_img_sigma, per_img_rec, per_img_sigma_ch
    """
    lps = max(1, patch_size // 8)
    sigma_mean_list = []
    sigma_ch_list   = []
    rec_list        = []
    per_img_sigma   = []          # mean channel
    per_img_rec     = []
    per_img_sigma_ch = {c: [] for c in range(N_CHANNELS)}  # 各通道逐图像列表

    for item in latent_cache:
        if item is None:
            continue
        sigma_np, re_np = item
        _, H_l, W_l = re_np.shape
        if H_l < lps or W_l < lps:
            continue
        try:
            ws_m, wr = compute_patch_stats_mean(sigma_np, re_np, patch_size)
            ws_m = np.asarray(ws_m).ravel()
            wr   = np.asarray(wr).ravel()
            if ws_m.size == 0:
                continue
            sigma_mean_list.append(ws_m)
            per_img_sigma.append(ws_m)
            per_img_rec.append(wr)

            ch_sigma, _ = compute_patch_stats_per_channel(sigma_np, re_np, patch_size)
            sigma_ch_list.append(ch_sigma)
            rec_list.append(wr)

            # 收集每个通道的逐图像 patch 数组
            for c in range(N_CHANNELS):
                per_img_sigma_ch[c].append(ch_sigma[c])   

        except Exception:
            continue

    if not rec_list:
        return None, None, None, None, None, None   

    sigma_mean = np.concatenate(sigma_mean_list)
    sigma_ch   = np.concatenate(sigma_ch_list, axis=1)
    rec_all    = np.concatenate(rec_list)
    return sigma_mean, sigma_ch, rec_all, per_img_sigma, per_img_rec, per_img_sigma_ch


def extract_image_data(latent_cache: list, patch_size: int):
    """聚合图像级：每张图的 σ 均值和重建误差均值"""
    lps = max(1, patch_size // 8)
    img_sigma_mean = []
    img_sigma_ch   = []
    img_rec        = []

    for item in latent_cache:
        if item is None:
            continue
        ls_np, re_np = item
        _, H_l, W_l  = re_np.shape
        if H_l < lps or W_l < lps:
            continue
        try:
            ws_m, wr = compute_patch_stats_mean(ls_np, re_np, patch_size)
            ws_m = np.asarray(ws_m).ravel()
            wr   = np.asarray(wr).ravel()
            if ws_m.size == 0:
                continue
            img_sigma_mean.append(float(ws_m.mean()))
            img_rec.append(float(wr.mean()))

            ch_sigma, _ = compute_patch_stats_per_channel(ls_np, re_np, patch_size)
            img_sigma_ch.append(ch_sigma.mean(axis=1))
        except Exception:
            continue

    if not img_rec:
        return None, None, None

    return (np.array(img_sigma_mean),
            np.array(img_sigma_ch).T,
            np.array(img_rec))


def compute_stats(x: np.ndarray, y: np.ndarray):
    """
    计算一对 (σ, rec_error) 的完整统计量。
    返回字典，包含 Spearman ρ, Pearson r, 回归参数, CI 带数组等。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]

    _nan_result = {
        'rho_s': float('nan'), 'p_s':   float('nan'),
        'r_p'  : float('nan'), 'p_p':   float('nan'),
        'slope': float('nan'), 'intercept': float('nan'),
        'r_sq' : float('nan'), 'n'     : int(len(x)),
        'x_range': np.array([]), 'y_fit': np.array([]),
        'ci_lo': np.array([]),   'ci_hi': np.array([]),
        'degenerate': True,
    }
    if len(x) < 10:
        return _nan_result
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        _nan_result['degenerate_reason'] = (
            'zero-variance x' if np.ptp(x) == 0 else 'zero-variance y'
        )
        return _nan_result

    MAX_STAT = 200_000
    if len(x) > MAX_STAT:
        idx   = np.random.choice(len(x), MAX_STAT, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y

    rho_s, p_s = spearmanr(xs, ys)
    r_p,   p_p = pearsonr(xs, ys)

    slope, intercept, r_val, p_lin, se = stats.linregress(xs, ys)

    x_range = np.linspace(xs.min(), xs.max(), 200)
    y_fit   = slope * x_range + intercept
    n       = len(xs)
    x_mean  = xs.mean()
    t_crit  = stats.t.ppf(0.975, df=n - 2)
    ss_x    = np.sum((xs - x_mean) ** 2)
    s_resid = se * np.sqrt(ss_x)
    se_band = s_resid * np.sqrt(1.0 / n + (x_range - x_mean) ** 2 / (ss_x + 1e-12))
    ci_lo   = y_fit - t_crit * se_band
    ci_hi   = y_fit + t_crit * se_band

    return {
        'rho_s'    : float(rho_s),  'p_s'      : float(p_s),
        'r_p'      : float(r_p),    'p_p'      : float(p_p),
        'slope'    : float(slope),  'intercept': float(intercept),
        'r_sq'     : float(r_val ** 2),
        'n'        : int(n),
        'x_range'  : x_range,
        'y_fit'    : y_fit,
        'ci_lo'    : ci_lo,
        'ci_hi'    : ci_hi,
        'degenerate': False,
    }


N_BINS_STRAT     = 10
N_BOOT_DEFAULT   = 200
N_SUBSAMPLE_BOOT = 20_000


def _run_decompose(per_img_sigma: list, per_img_rec: list):
    """
    调用 decompose_sigma 并返回清洗后的数组。
    进行全局均值中心化，返回 s_abs, s_img, s_rel, s_between, rec 等。
    """
    if not per_img_sigma:
        return None

    s_abs, s_img, s_rel, rec, rec_demeaned = decompose_sigma(per_img_sigma, per_img_rec)

    valid = (np.isfinite(s_rel) & np.isfinite(rec)
             & np.isfinite(s_img) & np.isfinite(rec_demeaned)) 
    if valid.sum() < 30:
        return None

    s_abs        = s_abs[valid];       s_img        = s_img[valid]
    s_rel        = s_rel[valid];       rec          = rec[valid]
    rec_demeaned = rec_demeaned[valid]             

    sigma_global = float(s_abs.mean())
    s_between    = s_img - sigma_global

    return {
        's_abs'         : s_abs,
        's_img'         : s_img,
        's_rel'         : s_rel,
        's_between'     : s_between,
        'sigma_global'  : sigma_global,
        'rec'           : rec,
        'rec_demeaned'  : rec_demeaned,   
        '_per_img_sigma': per_img_sigma,
        '_per_img_rec'  : per_img_rec,
    }


def _stratified_spearman(s_rel: np.ndarray, rec: np.ndarray,
                          s_img: np.ndarray, n_bins: int = N_BINS_STRAT):
    """
    分层 Spearman：按 σ_img 分位数分箱，箱内计算 ρ(σ_rel, rec)。
    返回加权平均 ρ、加权标准差、各箱 ρ 列表。
    """
    edges      = np.percentile(s_img, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-10

    rhos, weights = [], []
    for i in range(n_bins):
        mask = (s_img >= edges[i]) & (s_img < edges[i + 1]) \
               & np.isfinite(s_rel) & np.isfinite(rec)
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(s_rel[mask], rec[mask])
        if np.isfinite(rho):
            rhos.append(float(rho))
            weights.append(int(mask.sum()))

    if not rhos:
        return float('nan'), float('nan'), []

    w         = np.array(weights, dtype=float) / sum(weights)
    rho_wmean = float(np.dot(w, rhos))
    rho_wstd  = float(np.sqrt(np.dot(w, (np.array(rhos) - rho_wmean) ** 2)))
    return rho_wmean, rho_wstd, rhos


def _bootstrap_ci(decomposed: dict,
                   n_boot: int      = N_BOOT_DEFAULT,
                   n_subsample: int = N_SUBSAMPLE_BOOT,
                   seed: int        = 42):
    """
    图像级自助法估计 95% CI（对于原始 ρ、直接 ρ_rel、部分 ρ）。
    返回 (boot_raw, boot_direct, boot_partial)
    """
    per_img_sigma = decomposed['_per_img_sigma']
    per_img_rec   = decomposed['_per_img_rec']
    n_imgs = len(per_img_sigma)
    rng    = np.random.default_rng(seed)

    boot_raw, boot_direct, boot_partial = [], [], []

    for _ in range(n_boot):
        idx   = rng.integers(0, n_imgs, size=n_imgs)
        b_sig = [per_img_sigma[i] for i in idx]
        b_rec = [per_img_rec[i]   for i in idx]

        dec = _run_decompose(b_sig, b_rec)
        if dec is None:
            continue

        s_abs_b = dec['s_abs'];  s_rel_b = dec['s_rel']
        s_img_b = dec['s_img'];  rec_b   = dec['rec']

        if len(s_rel_b) > n_subsample:
            sel     = rng.choice(len(s_rel_b), n_subsample, replace=False)
            s_abs_b = s_abs_b[sel];  s_rel_b = s_rel_b[sel]
            s_img_b = s_img_b[sel];  rec_b   = rec_b[sel]

        rho_raw,    _ = spearmanr(s_abs_b, rec_b)
        rho_direct, _ = spearmanr(s_rel_b, rec_b)

        nb      = len(s_rel_b)
        r_srel  = rankdata(s_rel_b).astype(np.float64) / nb
        r_rec_b = rankdata(rec_b  ).astype(np.float64) / nb
        r_simg  = rankdata(s_img_b).astype(np.float64) / nb
        sl1, ic1 = np.polyfit(r_simg, r_srel, 1)
        sl2, ic2 = np.polyfit(r_simg, r_rec_b, 1)
        rho_partial, _ = spearmanr(r_srel  - (sl1 * r_simg + ic1),
                                    r_rec_b - (sl2 * r_simg + ic2))

        if all(np.isfinite([rho_raw, rho_direct, rho_partial])):
            boot_raw.append(float(rho_raw))
            boot_direct.append(float(rho_direct))
            boot_partial.append(float(rho_partial))

    return (np.array(boot_raw), np.array(boot_direct), np.array(boot_partial))


def compute_partial_stats(decomposed: dict,
                           n_boot: int = N_BOOT_DEFAULT) -> dict:
    """
    综合部分相关分析（含 4 项增强）：
    - 分层部分 ρ（真实非参数）
    - 交互效应指标（各箱 ρ 的标准差）
    - 图像级自助法 CI
    - 自助法 p 值
    """
    _nan = float('nan')
    _empty = dict(
        partial_rho=_nan, partial_r_sq=_nan,
        partial_ci95=[_nan, _nan], partial_p=_nan,
        partial_rho_bin_std=_nan, per_bin_rhos=[],
        raw_rho=_nan,   raw_p=_nan,   raw_ci95=[_nan, _nan],
        direct_rho_rel=_nan, direct_p_rel=_nan, direct_ci95=[_nan, _nan],
        rho_between=_nan, rho_between_p=_nan, sigma_global=_nan,
        confound_frac=_nan, confound_frac_ci95=[_nan, _nan], confound_frac_p=_nan,
        n_decomposed=0, n_boot_success=0, degenerate=True,
        rho_within   = _nan, r2_within    = _nan, r2_between   = _nan,
        r2_raw_pearson = _nan, f_local      = _nan,
    )

    if decomposed is None:
        return _empty

    s_abs     = decomposed['s_abs']
    s_rel     = decomposed['s_rel']
    rec_demeaned = decomposed['rec_demeaned']
    s_img     = decomposed['s_img']
    s_between = decomposed['s_between']
    rec       = decomposed['rec']
    sigma_global = decomposed['sigma_global']

    MAX_N = 200_000
    if len(s_rel) > MAX_N:
        idx       = np.random.default_rng(42).choice(len(s_rel), MAX_N, replace=False)
        s_abs     = s_abs[idx];     s_rel     = s_rel[idx]
        s_img     = s_img[idx];     s_between = s_between[idx]
        rec       = rec[idx]
        rec_demeaned = rec_demeaned[idx]

    n = len(s_rel)
    if n < 30:
        _empty['n_decomposed'] = int(n)
        return _empty

    partial_rho, partial_bin_std, per_bin_rhos = _stratified_spearman(
        s_rel, rec, s_img, N_BINS_STRAT
    )

    raw_rho,     raw_p     = spearmanr(s_abs, rec)
    direct_rho,  direct_p  = spearmanr(s_rel, rec)
    rho_between, rho_btn_p = spearmanr(s_between, rec)
    rho_within,  p_within   = spearmanr(s_rel, rec_demeaned)
    r_within,    _          = pearsonr(s_rel, rec_demeaned)
    r_between,   _          = pearsonr(s_img, rec)
    r_raw_p,     _          = pearsonr(s_abs, rec)
    r2_within    = float(r_within**2)
    r2_between   = float(r_between**2)
    r2_raw_p     = float(r_raw_p**2)
    f_local      = (r2_within / r2_raw_p
                if r2_raw_p > 1e-8 else _nan)

    def _safe_frac(rr, dr):
        return (rr - dr) / abs(rr) * 100.0 if abs(rr) > 1e-6 else _nan

    confound_frac = _safe_frac(float(raw_rho), float(direct_rho))

    boot_raw, boot_direct, boot_partial = _bootstrap_ci(decomposed, n_boot=n_boot)
    n_boot_ok = int(len(boot_raw))

    def _ci95(arr):
        return ([float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
                if len(arr) >= 10 else [_nan, _nan])

    boot_frac = np.array([_safe_frac(r, d)
                           for r, d in zip(boot_raw, boot_direct)])
    boot_frac = boot_frac[np.isfinite(boot_frac)]

    def _two_sided_p(boot_arr, observed):
        if len(boot_arr) < 10 or not np.isfinite(observed):
            return _nan
        if observed >= 0:
            return min(1.0, 2.0 * float(np.mean(boot_arr < 0)))
        else:
            return min(1.0, 2.0 * float(np.mean(boot_arr > 0)))

    partial_p_boot    = _two_sided_p(boot_partial, partial_rho)
    confound_p_boot   = (float(np.mean(boot_frac <= 0))
                         if len(boot_frac) >= 10 else _nan)

    return dict(
        partial_rho          = float(partial_rho) if np.isfinite(partial_rho) else _nan,
        partial_r_sq         = float(partial_rho**2) if np.isfinite(partial_rho) else _nan,
        partial_ci95         = _ci95(boot_partial),
        partial_p            = partial_p_boot,
        partial_rho_bin_std  = float(partial_bin_std) if np.isfinite(partial_bin_std) else _nan,
        per_bin_rhos         = [float(r) for r in per_bin_rhos],
        rho_between          = float(rho_between),
        rho_between_p        = float(rho_btn_p),
        sigma_global         = float(sigma_global),
        raw_rho              = float(raw_rho),
        raw_p                = float(raw_p),
        raw_ci95             = _ci95(boot_raw),
        direct_rho_rel       = float(direct_rho),
        direct_p_rel         = float(direct_p),
        direct_ci95          = _ci95(boot_direct),
        confound_frac        = confound_frac,
        confound_frac_ci95   = (_ci95(boot_frac) if len(boot_frac) >= 10
                                else [_nan, _nan]),
        confound_frac_p      = confound_p_boot,
        n_decomposed         = int(n),
        n_boot_success       = n_boot_ok,
        degenerate           = False,
        rho_within      = float(rho_within) if np.isfinite(rho_within) else _nan,
        r2_within       = r2_within,        
        r2_between      = r2_between,        
        r2_raw_pearson  = r2_raw_p,          
        f_local         = f_local,
    )


def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.25,
                  n_pts: int = 80):
    """LOWESS 平滑（若 statsmodels 缺失则退化为分箱均值）"""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xmax_pts = min(len(x), 20_000)
        idx  = np.random.choice(len(x), xmax_pts, replace=False)
        res  = lowess(y[idx], x[idx], frac=frac, return_sorted=True)
        return res[:, 0], res[:, 1]
    except ImportError:
        edges   = np.percentile(x, np.linspace(0, 100, n_pts + 1))
        x_mid, y_mid = [], []
        for i in range(n_pts):
            mask = (x >= edges[i]) & (x < edges[i + 1])
            if mask.sum() > 3:
                x_mid.append(x[mask].mean())
                y_mid.append(y[mask].mean())
        return np.array(x_mid), np.array(y_mid)


def density_color(x: np.ndarray, y: np.ndarray, subsample: int = 5000):
    """计算每个点的 KDE 密度，用于散点着色"""
    if len(x) > subsample:
        idx  = np.random.choice(len(x), subsample, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y

    try:
        kde    = gaussian_kde(np.vstack([xs, ys]))
        z      = kde(np.vstack([xs, ys]))
        order  = np.argsort(z)
        return xs[order], ys[order], z[order]
    except Exception:
        return xs, ys, np.ones(len(xs))


def _annotate_stats(ax, st: dict, fontsize: float = 7.5):
    """在坐标轴上添加统计信息文本框"""
    if st.get('degenerate', False):
        reason = st.get('degenerate_reason', 'insufficient data')
        txt = f"ρ = N/A\n({reason})\nn = {st.get('n', 0):,}"
    else:
        txt = (
            f"Spearman ρ = {st['rho_s']:+.3f} {pstars(st['p_s'])}\n"
            f"Pearson  r = {st['r_p']:+.3f} {pstars(st['p_p'])}\n"
            f"R² = {st['r_sq']:.3f}   n = {st['n']:,}"
        )
    at = AnchoredText(txt, loc='upper left', frameon=True,
                      pad=0.40, borderpad=0.45,
                      prop=dict(size=fontsize, family='monospace'))
    at.patch.set(boxstyle='round,pad=0.35', alpha=0.92,
                 facecolor='white', edgecolor='0.70', linewidth=0.8)
    ax.add_artist(at)


def draw_patch_panel(ax, x: np.ndarray, y: np.ndarray,
                     cmap: str, color: str, patch_size: int,
                     sigma_label: str, show_ylabel: bool = True):
    """
    绘制一个 patch 级子图：hexbin + 回归线 + CI 带 + LOWESS
    返回统计字典
    """
    if len(x) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='0.5')
        return {}

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]
    if len(x) == 0:
        ax.text(0.5, 0.5, 'No valid data\n(all NaN/Inf filtered)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color='crimson')
        return {'degenerate': True, 'n': 0}

    hb_probe = ax.hexbin(x, y, gridsize=60, mincnt=1,
                         cmap=cmap, linewidths=0.0, alpha=0.0)
    counts    = hb_probe.get_array()
    hb_probe.remove()

    pos_counts = counts[counts > 0]
    if len(pos_counts) >= 2 and pos_counts.min() < pos_counts.max():
        norm       = LogNorm(vmin=max(float(pos_counts.min()), 1),
                             vmax=float(pos_counts.max()))
        norm_label = 'Count (log)'
    else:
        norm       = None
        norm_label = 'Count'

    hb = ax.hexbin(x, y, gridsize=60, mincnt=1,
                   cmap=cmap, norm=norm,
                   linewidths=0.0, alpha=0.92)

    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='3.5%', pad=0.05)
    cb      = ax.get_figure().colorbar(hb, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(norm_label, fontsize=6.5, labelpad=2)

    st = compute_stats(x, y)

    if not st.get('degenerate', True) and len(st['x_range']) > 0:
        ax.fill_between(st['x_range'], st['ci_lo'], st['ci_hi'],
                        color=REG_COLOR, alpha=CI_ALPHA, zorder=3,
                        label='95% CI band')
        ax.plot(st['x_range'], st['y_fit'], '-', color=REG_COLOR,
                lw=1.5, zorder=4, label='Linear fit')

        try:
            lx, ly = lowess_smooth(x, y)
            ax.plot(lx, ly, '-', color=LOWESS_COLOR,
                    lw=1.6, zorder=5, alpha=0.85, label='LOWESS')
        except Exception:
            pass

    _annotate_stats(ax, st, fontsize=7.0)

    ax.set_title(f'{patch_size}px', fontsize=10, pad=5,
                 color=PS_COLORS[patch_size], fontweight='bold')
    ax.set_xlabel(f'{sigma_label}  (σ)', labelpad=4, fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel('Reconstruction Error (MSE)', labelpad=4, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return st


def draw_image_panel(ax, x: np.ndarray, y: np.ndarray,
                     color: str, patch_size: int,
                     sigma_label: str, show_ylabel: bool = True):
    """
    绘制一个图像级子图：密度着色散点 + KDE 轮廓 + 回归 + CI + LOWESS
    """
    if len(x) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='0.5')
        return {}

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]
    if len(x) == 0:
        ax.text(0.5, 0.5, 'No valid data\n(all NaN/Inf filtered)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=8, color='crimson')
        return {'degenerate': True, 'n': 0}

    xs, ys, zs = density_color(x, y)
    sc = ax.scatter(xs, ys, c=zs, cmap='viridis', s=22,
                    alpha=0.75, linewidths=0, zorder=3)
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='3.5%', pad=0.05)
    cb      = ax.get_figure().colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Density', fontsize=6.5, labelpad=2)

    try:
        if len(x) >= 10:
            kde   = gaussian_kde(np.vstack([x, y]))
            xg    = np.linspace(x.min(), x.max(), 80)
            yg    = np.linspace(y.min(), y.max(), 80)
            XX, YY = np.meshgrid(xg, yg)
            ZZ    = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contour(XX, YY, ZZ, levels=5, colors='0.4',
                       linewidths=0.6, alpha=0.55, zorder=4)
    except Exception:
        pass

    st = compute_stats(x, y)

    if not st.get('degenerate', True) and len(st['x_range']) > 0:
        ax.fill_between(st['x_range'], st['ci_lo'], st['ci_hi'],
                        color=REG_COLOR, alpha=CI_ALPHA + 0.05, zorder=2,
                        label='95% CI band')
        ax.plot(st['x_range'], st['y_fit'], '-', color=REG_COLOR,
                lw=1.8, zorder=5, label='Linear fit')

        try:
            if len(x) >= 10:
                lx, ly = lowess_smooth(x, y, frac=0.35)
                ax.plot(lx, ly, '-', color=LOWESS_COLOR,
                        lw=1.8, zorder=6, alpha=0.90, label='LOWESS')
        except Exception:
            pass

    _annotate_stats(ax, st, fontsize=7.5)

    ax.set_title(f'{patch_size}px', fontsize=10, pad=5,
                 color=PS_COLORS[patch_size], fontweight='bold')
    ax.set_xlabel(f'{sigma_label}  (mean σ per image)', labelpad=4, fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel('Mean Rec. Error per Image (MSE)', labelpad=4, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    y_span = np.ptp(y) if np.ptp(y) > 0 else 1.0
    ax.set_ylim(
        np.percentile(y, 0.5)  - 0.08 * y_span,
        np.percentile(y, 99.5) + 0.12 * y_span,
    )

    return st


def _make_legend_handles():
    """生成图例句柄"""
    return [
        Line2D([0], [0], color=REG_COLOR,    lw=1.5, label='Linear regression'),
        Line2D([0], [0], color=REG_COLOR,    lw=6,   alpha=CI_ALPHA, label='95% CI band'),
        Line2D([0], [0], color=LOWESS_COLOR, lw=1.6, label='LOWESS smooth'),
    ]


def render_patch_figure(all_patch_data: dict, sigma_key,
                        title: str, stem: str, sigma_label: str,
                        cmap: str, color: str, output_dir: str):
    """
    渲染一个 2×2 的 patch 级散点图（4 种 patch size）
    """
    patch_sizes = sorted(all_patch_data.keys())
    n_ps  = min(len(patch_sizes), 4)
    ncols = 2
    nrows = (n_ps + 1) // 2

    fig = plt.figure(figsize=(7.8 * ncols, 6.5 * nrows))
    fig.set_constrained_layout(False)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            left=0.08, right=0.96,
                            top=0.82, bottom=0.08,
                            hspace=0.38, wspace=0.55)

    all_stats = {}
    for idx, ps in enumerate(patch_sizes[:n_ps]):
        row, col = divmod(idx, ncols)
        ax       = fig.add_subplot(gs[row, col])
        data     = all_patch_data[ps]

        if sigma_key == 'mean':
            x = data['sigma_mean']
        else:
            x = data['sigma_ch'][sigma_key]
        y = data['rec']

        show_ylabel = (col == 0)
        st = draw_patch_panel(ax, x, y, cmap=cmap, color=color,
                              patch_size=ps, sigma_label=sigma_label,
                              show_ylabel=show_ylabel)
        all_stats[ps] = st

    fig.legend(handles=_make_legend_handles(),
               loc='upper right', bbox_to_anchor=(0.96, 0.90),
               fontsize=8, frameon=True, borderpad=0.5,
               labelspacing=0.35, handlelength=2.0)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.90)
    _save_scatter(fig, stem, output_dir)
    return all_stats


def render_image_figure(all_image_data: dict, sigma_key,
                        title: str, stem: str, sigma_label: str,
                        color: str, output_dir: str):
    """渲染一个 2×2 的图像级散点图"""
    patch_sizes = sorted(all_image_data.keys())
    n_ps  = min(len(patch_sizes), 4)
    ncols = 2
    nrows = (n_ps + 1) // 2

    fig = plt.figure(figsize=(7.8 * ncols, 6.5 * nrows))
    fig.set_constrained_layout(False)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            left=0.08, right=0.96,
                            top=0.82, bottom=0.08,
                            hspace=0.38, wspace=0.55)

    all_stats = {}
    for idx, ps in enumerate(patch_sizes[:n_ps]):
        row, col = divmod(idx, ncols)
        ax       = fig.add_subplot(gs[row, col])
        data     = all_image_data[ps]

        if sigma_key == 'mean':
            x = data['sigma_mean']
        else:
            x = data['sigma_ch'][sigma_key]
        y = data['rec']

        show_ylabel = (col == 0)
        st = draw_image_panel(ax, x, y, color=color,
                              patch_size=ps, sigma_label=sigma_label,
                              show_ylabel=show_ylabel)
        all_stats[ps] = st

    fig.legend(handles=_make_legend_handles(),
               loc='upper right', bbox_to_anchor=(0.96, 0.90),
               fontsize=8, frameon=True, borderpad=0.5,
               labelspacing=0.35, handlelength=2.0)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.90)
    _save_scatter(fig, stem, output_dir)
    return all_stats


def _save_scatter(fig: plt.Figure, stem: str, output_dir: str):
    """保存散点图到 output_dir/figures/scatter/ 目录"""
    out = Path(output_dir) / 'figures' / 'scatter'
    out.mkdir(parents=True, exist_ok=True)
    base = str(out / stem)
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    print(f"  [Saved] scatter/{stem}.png")


def render_partial_figure(all_decompose_data: dict, output_dir: str) -> dict:
    """
    绘制部分相关图：σ_rel (图像内偏差) vs 重建误差
    输出 scatter_partial_mean.png
    """
    patch_sizes = sorted(k for k, v in all_decompose_data.items() if v is not None)
    if not patch_sizes:
        print("  ⚠ [Partial] No decomposed data — skipping scatter_partial_mean.png")
        return {}

    n_ps  = min(len(patch_sizes), 4)
    ncols = 2
    nrows = (n_ps + 1) // 2

    fig = plt.figure(figsize=(7.8 * ncols, 6.5 * nrows))
    fig.set_constrained_layout(False)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            left=0.08, right=0.96,
                            top=0.80, bottom=0.08,
                            hspace=0.38, wspace=0.55)

    all_stats = {}
    for idx, ps in enumerate(patch_sizes[:n_ps]):
        row, col = divmod(idx, ncols)
        ax       = fig.add_subplot(gs[row, col])
        dec      = all_decompose_data[ps]

        show_ylabel = (col == 0)
        st = draw_patch_panel(
            ax, dec['s_rel'], dec['rec'],
            cmap        = 'GnBu',
            color       = C['teal'],
            patch_size  = ps,
            sigma_label = 'σ_rel  (within-image deviation)',
            show_ylabel = show_ylabel,
        )
        all_stats[ps] = st

    fig.legend(handles=_make_legend_handles(),
               loc='upper right', bbox_to_anchor=(0.96, 0.88),
               fontsize=8, frameon=True, borderpad=0.5,
               labelspacing=0.35, handlelength=2.0)
    fig.suptitle(
        'Partial  σ vs Reconstruction Error\n'
        r'$\sigma_\mathrm{rel} = \sigma_\mathrm{patch} - \bar{\sigma}_\mathrm{image}$'
        '   |   within-image deviation after removing between-image confound',
        fontsize=11, fontweight='bold', y=0.88,
    )
    _save_scatter(fig, 'scatter_partial_mean', output_dir)
    return all_stats


def render_partial_comparison_figure(all_decompose_stats: dict, output_dir: str):
    """
    双面板图：
      A: 原始 ρ / 直接 ρ_rel / 分层部分 ρ（带自助法 CI 和显著性星号）
      B: 图像级混杂分数 (ρ_raw − ρ_rel) / |ρ_raw| × 100%
    """
    patch_sizes = sorted(all_decompose_stats.keys())
    if not patch_sizes:
        print("  ⚠ [Partial] No decompose stats — skipping comparison figure.")
        return

    _nan = float('nan')
    raw_rho,    direct_rho,    partial_rho    = [], [], []
    raw_ci,     direct_ci,     partial_ci     = [], [], []
    partial_ps  = []
    fractions,  frac_ci,       frac_ps        = [], [], []

    for ps in patch_sizes:
        d = all_decompose_stats.get(ps, {})
        raw_rho.append(d.get('raw_rho',       _nan))
        raw_ci.append (d.get('raw_ci95',      [_nan, _nan]))
        direct_rho.append(d.get('direct_rho_rel', _nan))
        direct_ci.append (d.get('direct_ci95',    [_nan, _nan]))
        partial_rho.append(d.get('partial_rho',   _nan))
        partial_ci.append (d.get('partial_ci95',  [_nan, _nan]))
        partial_ps.append (d.get('partial_p',     _nan))
        fractions.append  (d.get('confound_frac', _nan))
        frac_ci.append    (d.get('confound_frac_ci95', [_nan, _nan]))
        frac_ps.append    (d.get('confound_frac_p',    _nan))

    x, w   = np.arange(len(patch_sizes)), 0.23
    labels  = [f'{ps}px' for ps in patch_sizes]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.set_constrained_layout(False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.64, bottom=0.13, wspace=0.38)

    def _errbar(ax_, xp, vals, cis, color='#111111'):
        lo = [max(0.0, v - ci[0]) if np.isfinite(v) and np.isfinite(ci[0]) else 0.0
              for v, ci in zip(vals, cis)]
        hi = [max(0.0, ci[1] - v) if np.isfinite(v) and np.isfinite(ci[1]) else 0.0
              for v, ci in zip(vals, cis)]
        valid = [i for i, v in enumerate(vals) if np.isfinite(v)]
        if not valid:
            return
        ax_.errorbar(xp[valid], [vals[i] for i in valid],
                     yerr=[[lo[i] for i in valid], [hi[i] for i in valid]],
                     fmt='none', color=color, capsize=3.5, lw=1.1,
                     capthick=1.1, zorder=10)

    def _pstars(ax_, xp, vals, cis, ps_arr, fontsize=8.5, gap_frac=0.06,
                neg_extra=0.0):
        for xi, v, ci, p in zip(xp, vals, cis, ps_arr):
            if not (np.isfinite(v) and np.isfinite(p)):
                continue
            s   = pstars(p)
            col = '#111111' if s != 'n.s.' else '0.60'
            gap = max(0.006, abs(v) * gap_frac)
            if v >= 0:
                y_ref = ci[1] if np.isfinite(ci[1]) else v
                ax_.text(xi, y_ref + gap, s, ha='center', va='bottom',
                         fontsize=fontsize, fontweight='bold', color=col)
            else:
                y_ref = ci[0] if np.isfinite(ci[0]) else v
                ax_.text(xi, y_ref - gap - neg_extra, s, ha='center', va='top',
                         fontsize=fontsize, fontweight='bold', color=col)

    def _vallabel_ci(ax_, bars, vals, cis, fmt='.3f',
                     gap=0.006, extra_offset=0.0, fontsize=6.5):
        for bar, v, ci in zip(bars, vals, cis):
            if not np.isfinite(v):
                continue
            h = bar.get_height()
            if h >= 0:
                ci_hi  = ci[1] if (ci and np.isfinite(ci[1])) else h
                y      = max(h, ci_hi) + gap + extra_offset
                va     = 'bottom'
            else:
                ci_lo  = ci[0] if (ci and np.isfinite(ci[0])) else h
                y      = min(h, ci_lo) - gap - extra_offset
                va     = 'top'
            ax_.text(bar.get_x() + bar.get_width() / 2, y,
                     f'{v:{fmt}}', ha='center', va=va,
                     fontsize=fontsize, fontfamily='monospace')

    b1 = ax.bar(x - w, raw_rho,    w, label='Raw ρ  (σ_abs vs rec)',
                color=C['blue'],   alpha=0.85, edgecolor='white', lw=0.6)
    b2 = ax.bar(x,     direct_rho, w, label='Direct ρ_rel  (σ_rel vs rec)',
                color=C['teal'],   alpha=0.85, edgecolor='white', lw=0.6)
    b3 = ax.bar(x + w, partial_rho, w,
                label='Partial ρ  (σ_rel | σ_img, stratified)',
                color=C['orange'], alpha=0.85, edgecolor='white', lw=0.6)

    _errbar(ax, x - w, raw_rho,    raw_ci)
    _errbar(ax, x,     direct_rho, direct_ci)
    _errbar(ax, x + w, partial_rho, partial_ci)

    _vallabel_ci(ax, b1, raw_rho,     raw_ci,     extra_offset=0.000)
    _vallabel_ci(ax, b2, direct_rho,  direct_ci,  extra_offset=0.000)
    _vallabel_ci(ax, b3, partial_rho, partial_ci, extra_offset=0.025)

    _pstars(ax, x + w, partial_rho, partial_ci, partial_ps, neg_extra=0.050)

    ax.axhline(0, color='0.45', lw=0.8, ls='--', zorder=0)
    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_xlabel('Patch Size', fontsize=10)
    ax.set_ylabel('Spearman ρ', fontsize=10)
    ax.set_title('A   Raw vs Partial Spearman ρ\n(error bars = 95% bootstrap CI)',
                 fontsize=11, fontweight='bold', pad=7)
    ax.legend(fontsize=7.5, framealpha=0.92, loc='lower left',
              labelspacing=0.28, borderpad=0.45)
    ax.text(0.99, 0.02,
            '* p<0.05   ** p<0.01\n*** p<0.001    n.s. p>0.05\n'
            '(bootstrap two-sided, H₀: partial ρ = 0)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color='0.35', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.80', lw=0.6, alpha=0.85))
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)

    all_A = [ci[k] for ci_l in [raw_ci, direct_ci, partial_ci]
             for ci in ci_l for k in (0, 1) if np.isfinite(ci[k])]
    if all_A:
        a_lo, a_hi = min(all_A), max(all_A)
        a_span     = max(a_hi - a_lo, 0.05)
        ax.set_ylim(a_lo - a_span * 0.35, a_hi + a_span * 0.35)

    bar_colors = [C['lpurple'] if np.isnan(f) or f >= 0 else C['lred']
                  for f in fractions]
    bars_f = ax2.bar(x, fractions, 0.48,
                     color=bar_colors, alpha=0.88, edgecolor='white', lw=0.6)

    all_y_B = ([f for f in fractions if np.isfinite(f)]
               + [ci[0] for ci in frac_ci if np.isfinite(ci[0])]
               + [ci[1] for ci in frac_ci if np.isfinite(ci[1])])
    if all_y_B:
        b_lo   = min(all_y_B)
        b_hi   = max(all_y_B)
        b_span = max(b_hi - b_lo, abs(b_lo) * 0.05, 1.0)
        ax2.set_ylim(b_lo - b_span * 0.30,
                     b_hi + b_span * 0.55)
    else:
        b_span = 10.0

    val_gap = b_span * 0.025
    for (bar, fv), ci in zip(zip(bars_f, fractions), frac_ci):
        if np.isfinite(fv):
            ci_top   = ci[1] if (ci and np.isfinite(ci[1])) else fv
            y_anchor = max(fv, ci_top) + val_gap
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     y_anchor,
                     f'{fv:+.1f}%', ha='center', va='bottom',
                     fontsize=7.5, fontfamily='monospace')

    _errbar(ax2, x, fractions, frac_ci)
    _pstars(ax2, x, fractions, frac_ci, frac_ps, fontsize=8.0, gap_frac=0.02)

    ax2.axhline(0, color='0.45', lw=0.8, ls='--', zorder=0)
    ax2.set_xticks(x);  ax2.set_xticklabels(labels)
    ax2.set_xlabel('Patch Size', fontsize=10)
    ax2.set_ylabel('ρ Change  (%)', fontsize=10)
    ax2.set_title(
        r'B   Image-level Confound Fraction'  '\n'
        r'$(ρ_\mathrm{raw} - ρ_\mathrm{rel})\;/\;|ρ_\mathrm{raw}|$',
        fontsize=11, fontweight='bold', pad=4,
    )

    all_pos = all(f >= 0 for f in fractions if np.isfinite(f))
    leg_loc = 'upper left' if all_pos else 'lower left'

    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color=C['lpurple'], alpha=0.88, label='Between-image confound > 0'),
        Patch(color=C['lred'],    alpha=0.88, label='Within-image effect stronger'),
    ], fontsize=8, framealpha=0.92, loc=leg_loc,
       labelspacing=0.30, borderpad=0.45)

    ax2.text(0.99, 0.02,
             '* p<0.05   ** p<0.01   *** p<0.001   n.s. p>0.05\n'
             '(bootstrap one-sided, H₀: confound fraction ≤ 0\n'
             ' i.e. between-image σ does NOT inflate raw ρ)',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=7, color='0.35', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.80', lw=0.6, alpha=0.85))
    ax2.spines['top'].set_visible(False);  ax2.spines['right'].set_visible(False)

    fig.suptitle(
        'decompose_sigma:  Image-level vs Within-image σ Contribution to Reconstruction Error',
        fontsize=12, fontweight='bold', y=0.97,
    )
    _save_scatter(fig, 'scatter_partial_comparison', output_dir)


def render_partial_bins_figure(all_decompose_stats: dict, output_dir: str):
    """
    交互效应图：每个 σ_img 分箱内的条件 ρ(σ_rel, rec)
    平坦线 → 无交互；有斜率 → σ_img 调节 σ_rel→rec 的关系
    """
    valid_ps = [k for k, v in all_decompose_stats.items()
                if v and not v.get('degenerate', True) and v.get('per_bin_rhos')]
    if not valid_ps:
        print("  ⚠ [Partial bins] No per-bin data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.14)

    for ps in sorted(valid_ps):
        bin_rhos = all_decompose_stats[ps]['per_bin_rhos']
        bin_std  = all_decompose_stats[ps].get('partial_rho_bin_std', float('nan'))
        n_bins   = len(bin_rhos)
        x_pct    = np.linspace(100 / (2 * n_bins), 100 - 100 / (2 * n_bins), n_bins)
        std_str  = f'{bin_std:.3f}' if np.isfinite(bin_std) else '?'
        ax.plot(x_pct, bin_rhos,
                marker=PS_MARKERS[ps], color=PS_COLORS[ps],
                lw=1.6, ms=5.5, alpha=0.85,
                label=f'{ps}px  (within-bin std = {std_str})')

    ax.axhline(0, color='0.45', lw=0.8, ls='--', zorder=0)
    ax.set_xlabel('σ_img Quantile Bin (percentile midpoint)', fontsize=10)
    ax.set_ylabel('Spearman ρ  (σ_rel vs rec | σ_img bin)', fontsize=10)
    ax.set_title(
        'Interaction Effect:  Conditional ρ(σ_rel, rec | σ_img)\n'
        'Flat → no interaction;  varying → σ_img moderates σ_rel → rec',
        fontsize=11, fontweight='bold', pad=7,
    )
    ax.legend(fontsize=8.5, framealpha=0.92, loc='lower left',
              labelspacing=0.35, borderpad=0.45)

    all_rhos = [r for ps in sorted(valid_ps)
                for r in all_decompose_stats[ps]['per_bin_rhos']
                if np.isfinite(r)]
    if all_rhos:
        p05, p95   = np.percentile(all_rhos, 5), np.percentile(all_rhos, 95)
        bulk_span  = max(p95 - p05, 0.05)
        y_lo_raw   = min(all_rhos)
        y_hi_raw   = max(all_rhos)
        y_lo_clip  = p05 - 2.0 * bulk_span
        clipped    = y_lo_raw < y_lo_clip
        y_lo_final = y_lo_clip if clipped else y_lo_raw - bulk_span * 0.12
        y_hi_final = y_hi_raw  + bulk_span * 0.12
        ax.set_ylim(y_lo_final, y_hi_final)

        if clipped:
            ax.text(0.99, 0.02,
                    f'⚠ Last bin clipped  (min ρ = {y_lo_raw:.3f})',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7, color='0.45', style='italic',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec='0.75', lw=0.5, alpha=0.80))

    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
    _save_scatter(fig, 'scatter_partial_bins', output_dir)


SIGMA_SPECS = [
    (0,      'Ch 0 σ',    'Channel 0'),
    (1,      'Ch 1 σ',    'Channel 1'),
    (2,      'Ch 2 σ',    'Channel 2'),
    (3,      'Ch 3 σ',    'Channel 3'),
    ('mean', 'Mean σ',    'Mean (all 4 ch)'),
]


def render_all_scatter(latent_cache: list, output_dir: str,
                       patch_sizes=None) -> dict:
    """
    为所有 patch size 构建数据，然后渲染所有散点图。
    返回嵌套的统计字典。
    """
    if patch_sizes is None:
        patch_sizes = Config.PATCH_SIZES

    print("\n[Scatter] Extracting patch-level and image-level data ...")
    all_patch_data      = {}
    all_image_data      = {}
    all_decompose_data  = {}
    all_decompose_stats = {}
    all_decompose_stats_ch = {}

    for ps in patch_sizes:
        print(f"  Patch size {ps}px ...")

        sm, sch, rec, per_img_sm, per_img_rec, per_img_sm_ch = extract_patch_data(latent_cache, ps)
        if sm is None:
            print(f"    ⚠ Skipping PS={ps}px (no data)")
            continue
        all_patch_data[ps] = {'sigma_mean': sm, 'sigma_ch': sch, 'rec': rec}
        print(f"    Patch-level: {len(sm):,} patches")

        dec = _run_decompose(per_img_sm, per_img_rec)
        all_decompose_data[ps]  = dec
        all_decompose_stats[ps] = compute_partial_stats(dec)
        all_decompose_stats_ch[ps] = {}
        for c in range(N_CHANNELS):
            dec_c = _run_decompose(per_img_sm_ch[c], per_img_rec)
            all_decompose_stats_ch[ps][c] = compute_partial_stats(dec_c)
            if not all_decompose_stats_ch[ps][c].get('degenerate', True):
                rw = all_decompose_stats_ch[ps][c]['rho_within']
                fl = all_decompose_stats_ch[ps][c]['f_local']
                print(f"    [Ch{c} biwi]  ρ_within = {rw:+.4f}   f_local = {fl:.1%}")
        if not all_decompose_stats[ps].get('degenerate', True):
            p_rho = all_decompose_stats[ps]['partial_rho']
            d_rho = all_decompose_stats[ps]['direct_rho_rel']
            print(f"    decompose_sigma:  partial ρ = {p_rho:+.4f}   "
                  f"direct ρ_rel = {d_rho:+.4f}")

        im, ich, irec = extract_image_data(latent_cache, ps)
        if im is None:
            continue
        all_image_data[ps] = {'sigma_mean': im, 'sigma_ch': ich, 'rec': irec}
        print(f"    Image-level: {len(im)} images")

    if not all_patch_data:
        print("  ⚠ No data available. Check latent cache.")
        return {}

    all_stats = {
        'patch'          : {},
        'image'          : {},
        'decompose'      : all_decompose_stats,
        'decompose_ch'    : all_decompose_stats_ch,
        'partial_scatter': {},
    }

    print("\n[Scatter] Rendering 10 patch + image figures ...")

    for sigma_key, label_short, label_long in SIGMA_SPECS:
        if sigma_key == 'mean':
            stem  = 'scatter_patch_mean'
            cmap  = SCATTER_CMAPS['mean']
            color = C['blue']
        else:
            stem  = f'scatter_patch_ch{sigma_key}'
            cmap  = SCATTER_CMAPS[sigma_key]
            color = CH_COLORS[sigma_key]

        title = (
            f'Patch-level  σ vs Reconstruction Error\n'
            f'σ source: {label_long}  |  each point = one latent patch'
        )
        st = render_patch_figure(
            all_patch_data, sigma_key,
            title=title, stem=stem,
            sigma_label=label_short, cmap=cmap, color=color,
            output_dir=output_dir,
        )
        all_stats['patch'][sigma_key] = st

    for sigma_key, label_short, label_long in SIGMA_SPECS:
        if sigma_key == 'mean':
            stem  = 'scatter_img_mean'
            color = C['blue']
        else:
            stem  = f'scatter_img_ch{sigma_key}'
            color = CH_COLORS[sigma_key]

        title = (
            f'Image-level  σ vs Reconstruction Error\n'
            f'σ source: {label_long}  |  each point = one image (patch-averaged)'
        )
        st = render_image_figure(
            all_image_data, sigma_key,
            title=title, stem=stem,
            sigma_label=label_short, color=color,
            output_dir=output_dir,
        )
        all_stats['image'][sigma_key] = st

    print("\n[Scatter] Rendering partial correlation figures ...")
    partial_scatter_stats = render_partial_figure(all_decompose_data, output_dir)
    all_stats['partial_scatter'] = partial_scatter_stats

    render_partial_comparison_figure(all_decompose_stats, output_dir)
    render_partial_bins_figure(all_decompose_stats, output_dir)

    print(f"\n[Scatter] All figures saved to {output_dir}/figures/scatter/")
    return all_stats


def _save_full_stats_json(all_stats: dict, output_dir: str):
    """
    保存完整的统计 JSON（含元数据头，patch/image/decompose 三部分）
    """
    from sigma_core import Config, _to_serial

    _STAT_KEEP = {'rho_s', 'p_s', 'r_p', 'p_p',
                  'slope', 'intercept', 'r_sq', 'n', 'degenerate'}

    _DECOMP_KEEP = {
        'partial_rho', 'partial_r_sq', 'partial_ci95', 'partial_p',
        'partial_rho_bin_std', 'per_bin_rhos',
        'rho_between', 'rho_between_p', 'sigma_global',
        'raw_rho', 'raw_p', 'raw_ci95',
        'direct_rho_rel', 'direct_p_rel', 'direct_ci95',
        'confound_frac', 'confound_frac_ci95', 'confound_frac_p',
        'n_decomposed', 'n_boot_success', 'degenerate',
        'rho_within', 'r2_within', 'r2_between',
        'r2_raw_pearson', 'f_local',
    }

    def _filter(st: dict, keys: set) -> dict:
        return _to_serial({k: v for k, v in st.items() if k in keys})

    output = {
        'vae_id'  : Config.VAE_ID,
        'img_type': Config.IMG_TYPE,
        'log'     : Config.LOG,
    }

    for level in ('patch', 'image'):
        output[level] = {}
        for sigma_key, stats_by_ps in all_stats.get(level, {}).items():
            sk_str = str(sigma_key)
            output[level][sk_str] = {}
            for ps, st in stats_by_ps.items():
                output[level][sk_str][str(ps)] = _filter(st, _STAT_KEEP)

    output['decompose'] = {
        str(ps): _filter(st, _DECOMP_KEEP)
        for ps, st in all_stats.get('decompose', {}).items()
    }

    output['decompose_ch'] = {}
    for c in range(N_CHANNELS):                   
        output['decompose_ch'][str(c)] = {         
            str(ps): _filter(
                all_stats.get('decompose_ch', {}).get(ps, {}).get(c, {}),
                _DECOMP_KEEP
            )
            for ps in all_stats.get('decompose_ch', {})
        }

    data_dir = Path(output_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / 'scatter_stats.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  [JSON] scatter_stats.json → {out_path}")
    print(f"         vae_id={Config.VAE_ID!r}  "
          f"img_type={Config.IMG_TYPE!r}  log={Config.LOG}")


def main():
    p = make_parser('Scatter plot: σ vs reconstruction error (patch & image level)')
    p.add_argument('--cache_pkl', default=None,
                   help='Path to pre-saved latent_cache.pkl (skips VAE inference)')
    args = p.parse_args()

    Config.apply_args(args)
    Config.make_dirs()

    if args.cache_pkl and Path(args.cache_pkl).exists():
        import pickle
        print(f"Loading latent cache from {args.cache_pkl} ...")
        with open(args.cache_pkl, 'rb') as f:
            latent_cache = pickle.load(f)
    else:
        img_paths    = discover_images(Config.IMG_DIR, Config.N_IMAGES)
        latent_cache = build_latent_cache(img_paths)
        cache_out    = Path(Config.OUTPUT_DIR) / 'data' / 'latent_cache.pkl'
        import pickle
        with open(cache_out, 'wb') as f:
            pickle.dump(latent_cache, f)
        print(f"Cache saved → {cache_out}")

    stats = render_all_scatter(latent_cache, Config.OUTPUT_DIR,
                               patch_sizes=Config.PATCH_SIZES)

    if stats:
        _RULE = '=' * 70
        print(f"\n{_RULE}")
        print("  Spearman ρ Summary")
        print(_RULE)
        ps_header = "  ".join(f"PS={p:2d}" for p in Config.PATCH_SIZES)
        print(f"  {'':20s}  {ps_header}")

        for level in ['patch', 'image']:
            print(f"\n  [{level.upper()} level]")
            for sigma_key, label_short, _ in SIGMA_SPECS:
                row_vals = []
                for ps in Config.PATCH_SIZES:
                    st  = stats.get(level, {}).get(sigma_key, {}).get(ps, {})
                    rho = st.get('rho_s', float('nan'))
                    row_vals.append(f"{rho:+.3f}")
                print(f"  {label_short:20s}  " + "  ".join(row_vals))

        print(f"\n  [PARTIAL ρ — decompose_sigma]")
        print(f"  {'':20s}  {ps_header}")
        for lbl, key in [('Partial ρ (σ_rel|σ_img)', 'partial_rho'),
                          ('Direct  ρ (σ_rel, rec)',  'direct_rho_rel')]:
            row_vals = []
            for ps in Config.PATCH_SIZES:
                dec_st = stats.get('decompose', {}).get(ps, {})
                rho    = dec_st.get(key, float('nan'))
                row_vals.append(f"{rho:+.3f}" if np.isfinite(rho) else "   N/A")
            print(f"  {lbl:20s}  " + "  ".join(row_vals))

        print(f"{_RULE}\n")

    _save_full_stats_json(stats, Config.OUTPUT_DIR)


if __name__ == '__main__':
    main()