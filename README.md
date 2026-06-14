# VAE 不确定性分析(当前仅包含mse代码)

本仓库提供了一套系统工具，用于分析预训练 VAE（例如 Stable Diffusion 的 VAE）中潜在不确定性（log‑sigma 或 sigma）与重建误差或感知损失LPIPS之间的关系。  
实现了 **patch 级** 与 **图像级** 的相关性分析，并提出了 **sigma 分解** 方法，将 sigma 分解为图像级分量和图像内偏差分量，从而分离混淆因素，量化偏相关效应。

## 🎯 主要功能

- **Patch 级散点图** – 每个潜在 patch 作为一个数据点（支持 8,16,32,64 px 四种 patch 尺寸）
- **图像级散点图** – 每张图像的聚合统计量
- **逐通道分析** – 分别处理 VAE 的 4 个潜在通道
- **Sigma 分解**（`decompose_sigma`）：  
  `σ_abs = σ_img + σ_rel`  
  其中 `σ_img` 是图像级的平均 sigma，`σ_rel` 是图像内部的偏差。
- **偏相关分析** – 通过分层 Spearman ρ(σ_rel, rec | σ_img) 消除图像间混淆
- **自助法置信区间**（95%）和置换检验风格的显著性评估
- **自动缓存** VAE 潜在特征，避免重复计算
- **完整统计输出**（JSON 格式）和可视化图片（PNG，300 DPI）

## 🖼️ 数据准备

将待分析的图像（支持 `.png, .jpg, .jpeg, .bmp, .tif, .webp`）放入一个文件夹中。  
默认路径为 `./LR`，可通过 `--img_dir` 参数修改。  
VAE 模型默认使用 `stabilityai/sd-vae-ft-ema`，会自动从 Hugging Face 下载并缓存到 `./vae_cache`。

## 🚀 使用方法

主要脚本为 `sigma_scatter_plots.py`，它会自动调用 `sigma_core.py` 中的核心功能。

### 基础运行

```bash
python sigma_scatter_plots.py --img_dir /path/to/your/images --output_dir ./results
```

### 常用参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--img_dir` | str | `./LR` | 输入图像文件夹路径 |
| `--output_dir` | str | `./verification_pub` | 输出根目录（图片和 JSON 将保存在其子目录中） |
| `--n_images` | int | `None` | 限制处理的图像数量（`None` 表示全部） |
| `--patch_sizes` | int list | `[8,16,32,64]` | 要分析的 patch 尺寸（像素） |
| `--n_perm` | int | `5000` | 置换检验迭代次数（用于某些检验） |
| `--n_bins` | int | `10` | 分层分析的分箱数 |
| `--seed` | int | `42` | 随机种子（保证可复现性） |
| `--img_type` | str | `LR` | 图像类型标签（写入 JSON 元数据） |
| `--log` | bool | `True` | 是否对 sigma 取对数（σ 实际是 log‑sigma） |
| `--cache_pkl` | str | `None` | 预先生成的 latent cache 文件路径（跳过 VAE 推理） |

### 运行示例

```bash
# 分析 100 张图像，只使用 16 和 32 两种 patch 尺寸
python sigma_scatter_plots.py --img_dir ./my_images --n_images 100 --patch_sizes 16 32 --output_dir ./exp1

# 使用已有的 cache（加速重复运行）
python sigma_scatter_plots.py --cache_pkl ./verification_pub/data/latent_cache.pkl
```

## 📁 输出结构

运行后会在 `output_dir` 下生成：

```
output_dir/
├── figures/
│   ├── scatter/                     # 所有散点图
│   │   ├── scatter_patch_mean.png
│   │   ├── scatter_patch_ch0.png
│   │   ├── scatter_img_mean.png
│   │   ├── scatter_partial_mean.png
│   │   ├── scatter_partial_comparison.png
│   │   └── scatter_partial_bins.png
│   └── ... (其他分类图)
├── data/
│   ├── latent_cache.pkl             # VAE 特征缓存（加速后续运行）
│   └── scatter_stats.json           # 完整统计结果（JSON）
```

### JSON 统计文件内容示例

```json
{
  "vae_id": "stabilityai/sd-vae-ft-ema",
  "img_type": "LR",
  "log": true,
  "patch": {
    "mean": { "8": {"rho_s": 0.452, "p_s": 0.0, "n": 15234, ...}, ... },
    "0": {...}, "1": {...}, "2": {...}, "3": {...}
  },
  "image": { ... },
  "decompose": {
    "8": {
      "partial_rho": 0.123,
      "partial_ci95": [0.098, 0.149],
      "direct_rho_rel": 0.089,
      "confound_frac": 35.2,
      ...
    }
  },
  "decompose_ch": { ... }
}
```

## 📊 结果解读

- **Patch 级图**：每个点代表一个 latent patch，颜色表示密度。显示线性回归线、95% 置信带和 LOWESS 平滑曲线。左上角显示 Spearman ρ、Pearson r、R² 和样本量。
- **图像级图**：每张图像聚合为一个点，展示整体趋势。
- **偏相关图**：`scatter_partial_mean.png` 展示 σ_rel（图像内偏差）与重建误差的关系，已剔除图像级混淆。
- **比较图**（`scatter_partial_comparison.png`）：
  - 左图：原始 ρ、直接 ρ_rel、分层偏 ρ（带自助法 CI）
  - 右图：图像级混淆分数 = (ρ_raw − ρ_rel) / |ρ_raw| × 100%  
    正值表示图像间 sigma 差异夸大了原始相关性；负值表示图像内效应更强。
- **交互效应图**（`scatter_partial_bins.png`）：按 σ_img 分箱后的条件 ρ(σ_rel, rec)，若曲线非平坦，则表明存在交互作用。

## 🧠 核心算法简述

1. **VAE 前向**：对每张图像计算后验分布 `log σ` 和重建误差（逐像素 MSE，下采样到 latent 空间尺寸）。
2. **Patch 提取**：将 latent 特征图划分为不重叠的 patch，计算每个 patch 内的平均 σ（或逐通道）和平均重建误差。
3. **Sigma 分解**：
   - `σ_img` = 图像内所有 patch 的 σ 均值
   - `σ_rel` = σ_patch − σ_img
4. **分层偏相关**：按 σ_img 分位数将数据分为 `n_bins` 箱，在每个箱内计算 ρ(σ_rel, rec)，然后加权平均得到偏相关估计。
5. **自助法**：对图像进行有放回重采样（图像级），估计各统计量的 95% 置信区间和双侧 p 值。
