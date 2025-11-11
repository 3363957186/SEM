#!/usr/bin/env python3
import h5py, numpy as np, pandas as pd, toml
from pathlib import Path
from collections import Counter

H5AD = "/Users/e/Downloads/1.h5ad"                 # 改成你的文件名
OUT_CSV = "./data/donor_features.csv"  # 输出表格
OUT_TOML = "stage_1.toml"       # 输出TOML
ID_COL = "donor_id"             # 行ID（样本ID）
# 你想汇总的 obs 分类列：会输出每个donor的“众数”值
STATUS_COLS = [
    "AD_status", "Tauopathy_status", "DLBD_status", "Vascular_status",
    "sex", "genetic_ancestry", "disease"
]
# 用来做组成特征的细胞类型列（二选一，优先更细的；若你想用 subclass/subtype，请替换）
CELLTYPE_KEY = "cell_type"  # 或 "subclass" / "subtype"

print(f"Opening {H5AD} ...")
with h5py.File(H5AD, "r") as f:
    # ---- 读 donor_id ----
    donors_cat = np.array(f["/obs/donor_id/categories"], dtype=str)
    donors_codes = np.array(f["/obs/donor_id/codes"])
    n_cells = donors_codes.shape[0]
    print(f"cells: {n_cells}, donors: {len(donors_cat)}")

    # ---- 读 cell type ----
    ct_cat = np.array(f[f"/obs/{CELLTYPE_KEY}/categories"], dtype=str)
    ct_codes = np.array(f[f"/obs/{CELLTYPE_KEY}/codes"])
    n_ct = len(ct_cat)
    print(f"cell types ({CELLTYPE_KEY}): {n_ct}")

    # ---- 组装一个 DataFrame 的索引：donor 名称 ----
    donor_names = donors_cat  # index->name
    # 为每个donor建桶
    n_donors = donor_names.size
    counts_total = np.zeros(n_donors, dtype=np.int64)
    counts_ct = np.zeros((n_donors, n_ct), dtype=np.int64)

    # 累加细胞到各donor、celltype
    # donors_codes / ct_codes 长度均为 n_cells，值是类别索引
    np.add.at(counts_total, donors_codes, 1)
    np.add.at(counts_ct, (donors_codes, ct_codes), 1)

    # ---- 计算比例 ----
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_ct = counts_ct / counts_total[:, None]
        frac_ct[np.isnan(frac_ct)] = 0.0

    # ---- 为每个donor计算状态列（众数/出现最多的类别）----
    status_data = {}
    for key in STATUS_COLS:
        if f"/obs/{key}/categories" not in f or f"/obs/{key}/codes" not in f:
            print(f"[WARN] skip {key}: not found")
            continue
        cat = np.array(f[f"/obs/{key}/categories"], dtype=str)
        codes = np.array(f[f"/obs/{key}/codes"])
        # 计算每个donor的众数
        mode_idx = np.full(n_donors, -1, dtype=np.int64)
        for d in range(n_donors):
            # 在该donor的细胞索引子集上取众数
            mask = (donors_codes == d)
            if not np.any(mask):
                continue
            vals = codes[mask]
            # 众数：频次最高的 code；若并列，取较小索引
            c = Counter(vals.tolist()).most_common(1)
            mode_idx[d] = c[0][0]
        status_data[key] = np.where(mode_idx >= 0, cat[mode_idx], pd.NA)

    # ---- 组织成DataFrame ----
    # 行：donor；列：总细胞数、各celltype计数与比例、状态列
    df = pd.DataFrame(index=donor_names)
    df.index.name = ID_COL
    df["n_cells"] = counts_total

    # 细胞类型计数与比例
    for j, ct in enumerate(ct_cat):
        df[f"count_{CELLTYPE_KEY}__{ct}"] = counts_ct[:, j]
    for j, ct in enumerate(ct_cat):
        df[f"frac_{CELLTYPE_KEY}__{ct}"] = frac_ct[:, j]

    # 状态列
    for k, v in status_data.items():
        df[k] = v

    # （可选）把缺失填充成明确字符串
    df[STATUS_COLS] = df[STATUS_COLS].astype("string")

    # 保存
    df.to_csv(OUT_CSV)
    print(f"Saved {OUT_CSV} with shape {df.shape}")

    # ---- 生成 TOML ----
    feature_types = {}
    for c in df.columns:
        if c == "n_cells" or c.startswith("count_") or c.startswith("frac_"):
            feature_types[c] = "numerical"
        else:
            feature_types[c] = "categorical"

    toml_dict = {
        "data": {"train_csv": OUT_CSV, "id_column": ID_COL},
        "features": feature_types
    }
    Path(OUT_TOML).write_text(toml.dumps(toml_dict), encoding="utf-8")
    print(f"Saved {OUT_TOML}")
