import os, warnings, random, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from scipy.stats import norm as spnorm
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置随机种子，保证可复现
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(42)

PUB_RC = {
    'figure.dpi'                    : 150,
    'savefig.dpi'                   : 300,
    'savefig.bbox'                  : 'tight',
    'savefig.pad_inches'            : 0.08,
    'font.family'                   : 'DejaVu Sans',
    'font.size'                     : 9,
    'axes.labelsize'                : 10,
    'axes.titlesize'                : 11,
    'axes.titleweight'              : 'bold',
    'axes.titlepad'                 : 8,
    'axes.labelpad'                 : 5,
    'xtick.labelsize'               : 8,
    'ytick.labelsize'               : 8,
    'xtick.direction'               : 'out',
    'ytick.direction'               : 'out',
    'xtick.major.size'              : 3.5,
    'ytick.major.size'              : 3.5,
    'xtick.major.pad'               : 3,
    'ytick.major.pad'               : 3,
    'legend.fontsize'               : 8,
    'legend.framealpha'             : 0.92,
    'legend.edgecolor'              : '0.75',
    'legend.borderpad'              : 0.5,
    'legend.labelspacing'           : 0.35,
    'axes.spines.top'               : False,
    'axes.spines.right'             : False,
    'axes.linewidth'                : 0.8,
    'xtick.major.width'             : 0.8,
    'ytick.major.width'             : 0.8,
    'lines.linewidth'               : 1.6,
    'patch.linewidth'               : 0.8,
    'grid.linewidth'                : 0.45,
    'grid.alpha'                    : 0.35,
    'errorbar.capsize'              : 3.5,
    'figure.constrained_layout.use' : True,
    'figure.constrained_layout.h_pad': 0.12,
    'figure.constrained_layout.w_pad': 0.12,
    'figure.constrained_layout.hspace': 0.05,
    'figure.constrained_layout.wspace': 0.05,
}
plt.rcParams.update(PUB_RC)

# 配色
C = {
    'blue'   : '#2166ac',  'red'    : '#d6604d',
    'green'  : '#4dac26',  'purple' : '#762a83',
    'orange' : '#e08214',  'teal'   : '#018571',
    'gray'   : '#888888',  'lblue'  : '#92c5de',
    'lred'   : '#f4a582',  'lgreen' : '#a1d99b',
    'lpurple': '#c2a5cf',  'lyellow': '#fee090',
    'brown'  : '#8c510a',  'pink'   : '#de77ae',
}

PS_COLORS  = {8: C['blue'], 16: C['green'], 32: C['red'], 64: C['purple']}
PS_MARKERS = {8: 'o', 16: 's', 32: '^', 64: 'D'}

CH_COLORS  = {0: C['blue'], 1: C['orange'], 2: C['green'], 3: C['red']}
CH_NAMES   = {0: 'Channel 0', 1: 'Channel 1', 2: 'Channel 2', 3: 'Channel 3'}
CH_MARKERS = {0: 'o', 1: 's', 2: '^', 3: 'D'}

TEST_NAMES_SHORT = ['Spearman', 'Permut.', 'Monoton.',
                    'MI', 'Pred R²', 'Partial ρ', 'Joint R²']
TEST_NAMES_FULL  = ['Test 1\nSpearman', 'Test 2\nPermut.', 'Test 3\nMonot.',
                    'Test 4\nMI', 'Test 5\nPred. R²', 'Test 6\nPartial ρ',
                    'Test 7\nJoint R²']


class Config:
    """全局配置，可通过命令行参数覆盖"""
    VAE_ID   = "stabilityai/sd-vae-ft-ema"
    IMG_TYPE = "LR"
    LOG      = True

    IMG_DIR             = "./LR"
    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
    PATCH_SIZES         = [8, 16, 32, 64]
    RANDOM_SEED         = 42
    VAE_CACHE_DIR       = "./vae_cache"
    OUTPUT_DIR          = "./verification_pub"
    N_IMAGES            = None
    N_PERMUTATIONS      = 5000
    N_BINS              = 10
    BOOTSTRAP_N         = 1000
    LATENT_SCALE_FACTOR = 8

    THRESHOLDS = {
        'rho_mean'         : 0.10,
        'n_positive_ratio' : 0.60,
        'p_perm'           : 0.001,
        'kendall_tau'      : 0.20,
        'effect_ratio'     : 1.10,
        'nmi'              : 0.01,
        'mi_p'             : 0.001,
        'r2'               : 0.02,
        'mae_improvement'  : 0.05,
        'partial_rho'      : 0.05,
    }

    @classmethod
    def apply_args(cls, args):
        """用命令行参数覆盖默认配置"""
        if hasattr(args, 'img_dir')    and args.img_dir:    cls.IMG_DIR        = args.img_dir
        if hasattr(args, 'output_dir') and args.output_dir: cls.OUTPUT_DIR     = args.output_dir
        if hasattr(args, 'n_images')   and args.n_images:   cls.N_IMAGES       = args.n_images
        if hasattr(args, 'patch_sizes')and args.patch_sizes: cls.PATCH_SIZES   = args.patch_sizes
        if hasattr(args, 'n_perm')     and args.n_perm:     cls.N_PERMUTATIONS = args.n_perm
        if hasattr(args, 'n_bins')     and args.n_bins:     cls.N_BINS         = args.n_bins
        if hasattr(args, 'seed')       and args.seed:
            cls.RANDOM_SEED = args.seed
            set_seed(args.seed)
        if hasattr(args, 'img_type') and args.img_type is not None:
            cls.IMG_TYPE = args.img_type
        if hasattr(args, 'log') and args.log is not None:
            cls.LOG = args.log

    @classmethod
    def make_dirs(cls):
        """创建输出子目录"""
        for sub in ['figures', 'data']:
            Path(cls.OUTPUT_DIR, sub).mkdir(parents=True, exist_ok=True)
        Path(cls.VAE_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def fisher_z_ci(rho_list: np.ndarray, alpha: float = 0.05):
    """用Fisher z变换计算Spearman相关系数的置信区间"""
    z   = np.arctanh(np.clip(rho_list, -0.9999, 0.9999))
    n   = len(z)
    zm  = z.mean()
    zse = z.std(ddof=1) / np.sqrt(n)
    zc  = spnorm.ppf(1 - alpha / 2)
    return float(np.tanh(zm - zc * zse)), float(np.tanh(zm + zc * zse)), float(np.tanh(zm))


def bootstrap_ci(arr: np.ndarray, statfn, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42):
    """基于百分位法的自助法置信区间"""
    rng  = np.random.default_rng(seed)
    boot = np.array([statfn(rng.choice(arr, len(arr), replace=True))
                     for _ in range(n_boot)])
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def bh_fdr(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR校正"""
    n   = len(pvalues)
    idx = np.argsort(pvalues)
    sp  = pvalues[idx]
    adj = np.minimum(sp * n / np.arange(1, n + 1), 1.0)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out       = np.empty(n)
    out[idx]  = adj
    return out


def pstars(p: float) -> str:
    """返回显著性标记（APA惯例）"""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def _to_serial(obj):
    """将numpy类型转换为Python原生类型，便于JSON序列化"""
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.bool_):     return bool(obj)
    if isinstance(obj, dict):         return {k: _to_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):return [_to_serial(i) for i in obj]
    return obj


def savefig(fig: plt.Figure, stem: str, out_dir: str, subdir: str = 'figures'):
    """保存图片为300dpi的PNG"""
    fig_dir = Path(out_dir) / subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    base = str(fig_dir / stem)
    fig.savefig(base + '.png', dpi=300)
    plt.close(fig)
    print(f"    [Figure] {subdir}/{stem}.png")


_VAE_INSTANCE = None

def get_vae() -> AutoencoderKL:
    """加载VAE模型（单例模式）"""
    global _VAE_INSTANCE
    if _VAE_INSTANCE is None:
        print(f"Loading VAE ({Config.VAE_ID}) in fp32 ...")
        _VAE_INSTANCE = AutoencoderKL.from_pretrained(
            Config.VAE_ID,
            subfolder="vae",
            cache_dir=Config.VAE_CACHE_DIR,
            local_files_only=False,
            torch_dtype=torch.float32,
        ).to(Config.DEVICE).eval()
        print(f"  VAE loaded → {Config.DEVICE} (fp32)")
    return _VAE_INSTANCE


def _crop8(im: Image.Image) -> Image.Image:
    """将图像裁剪到能被8整除的尺寸（VAE要求）"""
    w, h = im.size
    return im.crop((0, 0, (w // 8) * 8, (h // 8) * 8))


_to_tensor = transforms.ToTensor()

def prepare_image(img_path: str) -> Image.Image:
    """读取并预处理图像（RGB，裁剪到8的倍数）"""
    im = Image.open(img_path).convert('RGB')
    return _crop8(im)


def build_latent_cache(img_paths: list) -> list:
    """对每张图像做一次VAE前向，缓存log_sigma和重建误差（用于后续打patch）"""
    vae   = get_vae()
    cache = []
    for path in tqdm(img_paths, desc="Building latent cache"):
        try:
            img  = prepare_image(path)

            t_01 = _to_tensor(img).unsqueeze(0)          # [0,1] fp32
            t    = (t_01 * 2.0 - 1.0).to(Config.DEVICE) # [-1,1] fp32

            with torch.no_grad():
                posterior = vae.encode(t).latent_dist

                log_sigma = 0.5 * posterior.logvar       # log(σ), fp32, (1,4,H_l,W_l)
                mean_f    = posterior.mean

                z = mean_f + torch.exp(log_sigma) * torch.randn_like(mean_f)

                rec = vae.decode(z).sample

                mse     = (t - rec).pow(2).mean(dim=1, keepdim=True)
                H_lat   = log_sigma.shape[2]
                W_lat   = log_sigma.shape[3]
                rec_err = F.adaptive_avg_pool2d(mse, (H_lat, W_lat))

                ls_bad  = (torch.isnan(log_sigma).any() or
                           torch.isinf(log_sigma).any())
                rec_bad = (torch.isnan(rec_err).any() or
                           torch.isinf(rec_err).any())
                has_bad = ls_bad or rec_bad

            if has_bad:
                reason = ("log_sigma " if ls_bad else "") + ("rec_err" if rec_bad else "")
                print(f"  ⚠  {Path(path).name}: NaN/Inf in [{reason.strip()}] — skipped")
                cache.append(None)
                continue

            cache.append((
                log_sigma.squeeze(0).cpu().numpy(),   # (4, H_l, W_l)
                rec_err.squeeze(0).cpu().numpy(),     # (1, H_l, W_l)
            ))
        except Exception as e:
            print(f"  ⚠  {Path(path).name}: {e}")
            cache.append(None)
    valid = sum(1 for x in cache if x is not None)
    print(f"  Cached {valid}/{len(img_paths)} images.")
    return cache


def compute_patch_stats_mean(ls_np: np.ndarray, re_np: np.ndarray,
                              pixel_patch_size: int):
    """先将4个通道的σ取平均，再提取patch统计量（均值）"""
    lps = max(1, pixel_patch_size // 8)
    ls  = torch.from_numpy(ls_np).unsqueeze(0)         # (1, 4, H_l, W_l)
    re  = torch.from_numpy(re_np).unsqueeze(0)         # (1, 1, H_l, W_l)
    ls_mean = ls.mean(dim=1, keepdim=True)             # (1, 1, H_l, W_l)
    sp  = F.unfold(ls_mean, kernel_size=lps, stride=lps)
    rp  = F.unfold(re,      kernel_size=lps, stride=lps)
    return sp.mean(dim=1).squeeze().numpy(), rp.mean(dim=1).squeeze().numpy()


def compute_patch_stats_per_channel(ls_np: np.ndarray, re_np: np.ndarray,
                                     pixel_patch_size: int):
    """对每个通道独立提取patch统计量，返回 (4, N_patches) 和 (N_patches,)"""
    lps = max(1, pixel_patch_size // 8)
    ls  = torch.from_numpy(ls_np).unsqueeze(0)   # (1, 4, H_l, W_l)
    re  = torch.from_numpy(re_np).unsqueeze(0)   # (1, 1, H_l, W_l)

    ch_patches = []
    for c in range(4):
        lc = ls[:, c:c+1, :, :]
        sp = F.unfold(lc, kernel_size=lps, stride=lps)
        ch_patches.append(sp.mean(dim=1).squeeze().numpy())

    rp  = F.unfold(re, kernel_size=lps, stride=lps)
    rec = rp.mean(dim=1).squeeze().numpy()

    return np.array(ch_patches), rec


def decompose_sigma(w_sigma_per_image, w_rec_per_image):
    """将σ分解为图像级（σ_img）和patch级相对值（σ_rel），用于偏相关分析"""
    s_abs, s_img, s_rel, rec, rec_demeaned = [], [], [], [], []
    for ws, wr in zip(w_sigma_per_image, w_rec_per_image):
        mu_s = ws.mean()
        mu_r = wr.mean()                        #图像内误差均值
        s_abs.extend(ws)
        s_img.extend([mu_s] * len(ws))
        s_rel.extend(ws - mu_s)
        rec.extend(wr)
        rec_demeaned.extend(wr - mu_r)          
    return (np.array(s_abs), np.array(s_img),
            np.array(s_rel), np.array(rec),
            np.array(rec_demeaned))


def discover_images(img_dir: str, n_images=None) -> list:
    """递归查找目录中的所有图片，并可随机抽取子集"""
    exts  = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    paths = sorted([str(p) for p in Path(img_dir).rglob('*')
                    if p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    if n_images and n_images < len(paths):
        rng   = np.random.default_rng(Config.RANDOM_SEED)
        paths = list(rng.choice(paths, n_images, replace=False))
    print(f"  Discovered {len(paths)} images in '{img_dir}'")
    return paths


def _str2bool(v) -> bool:
    """解析布尔型命令行参数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true',  '1', 'yes', 'on'):  return True
    if v.lower() in ('false', '0', 'no',  'off'): return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got: {v!r}")


def make_parser(description: str = "VAE σ correlation analysis") -> argparse.ArgumentParser:
    """创建通用命令行解析器"""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--img_dir',    default='./LR',                help='Path to LR image folder')
    p.add_argument('--output_dir', default='./verification_pub',  help='Output root directory')
    p.add_argument('--n_images',   type=int,   default=None,      help='Max images to process (None=all)')
    p.add_argument('--patch_sizes',type=int,   nargs='+', default=[8, 16, 32, 64])
    p.add_argument('--n_perm',     type=int,   default=5000,      help='Permutation test iterations')
    p.add_argument('--n_bins',     type=int,   default=10,        help='Bins for conditional analysis')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--img_type',   default=None,
                   help='Image type tag written to JSON (e.g. LR, HR, SR, bicubic). '
                        'Default: Config.IMG_TYPE = "LR"')
    p.add_argument('--log',        type=_str2bool, default=None,
                   help='Whether σ is log-transformed; written to JSON. '
                        'Accepts true/false/1/0. Default: Config.LOG = True')
    return p