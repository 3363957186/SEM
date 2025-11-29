import csv
import os

import anndata as ad
import scipy.sparse as sp
from scipy.io import mmwrite
import pandas as pd

OUT_CSV = "mssm_cells_metadata.csv"

#adata = ad.read_h5ad("/Users/e/Downloads/1.h5ad")          # 常规读取
adata = ad.read_h5ad("/Users/e/Downloads/1.h5ad", backed="r")  # 超大文件只读内存映射

n = min(100_000, adata.n_obs)   # 防止总细胞数 < 10w
print("Subsetting to first", n, "cells out of", adata.n_obs)
adata = adata[:n, :].to_memory()

adata.obs.head()                                 # 细胞元数据（含 donor / 病理分期 等）
adata.var.head()                                 # 基因信息
list(adata.layers) if adata.layers is not None else None
adata.obsm.keys()                                # UMAP/PC 等嵌入

# 观测列名
print(adata.obs.columns.tolist())

# 1) 取 obs（所有你看到的列）并把 cell_id 放到第一列
obs = adata.obs.copy()
obs.insert(0, "cell_id", adata.obs_names.astype(str))

# 2) 可选：把 UMAP / tSNE 坐标也并过去，便于后续画图或筛选
if "X_umap" in adata.obsm:
    umap = pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs_names,
        columns=[f"UMAP{i+1}" for i in range(adata.obsm["X_umap"].shape[1])]
    )
    obs = obs.join(umap)
if "X_tsne" in adata.obsm:
    tsne = pd.DataFrame(
        adata.obsm["X_tsne"],
        index=adata.obs_names,
        columns=[f"tSNE{i+1}" for i in range(adata.obsm["X_tsne"].shape[1])]
    )
    obs = obs.join(tsne)

# 3) 避免分类类型在部分软件里显示异常：转成字符串更保险
for c in obs.select_dtypes(include=["category"]).columns:
    obs[c] = obs[c].astype(str)

# 4) 导出为 CSV（带 utf-8-sig 方便 Excel 识别中文；值里有逗号会自动加引号）
obs.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
print("Wrote", OUT_CSV, "with shape:", obs.shape)

# 在写 CSV 前后加：

out = os.path.abspath(OUT_CSV)  # 你的输出文件名
print(">>> Will write to:", out)

obs.to_csv(out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
print(">>> Wrote:", out, "rows:", len(obs), "cols:", obs.shape[1])

# 立刻用 pandas 回读确认行数（排除可视化工具只预览的情况）
df_chk = pd.read_csv(out)
print(">>> Re-read shape:", df_chk.shape, "(should be (", len(obs), ",", obs.shape[1], "))")
