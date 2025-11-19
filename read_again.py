# re_export.py
import os, csv
import anndata as ad
import pandas as pd

adata = ad.read_h5ad("your_file.h5ad")
obs = adata.obs.copy()
obs.insert(0, "cell_id", adata.obs_names.astype(str))
for c in obs.select_dtypes(include=["category"]).values:
    obs[c] = obs[c].astype(str)

# 强制每个字段加引号，统一换行符，避免解析差异
tmp = "cells_metadata.tmp.csv"
obs.to_csv(tmp, index=False, encoding="utf-8-sig",
           quoting=csv.QUOTE_ALL, lineterminator="\n")
os.replace(tmp, "cells_metadata_full.csv")
print("Wrote", os.path.abspath("cells_metadata_full.csv"), "rows:", len(obs))

# 可选：直接导 Excel，最不容易被误读
obs.to_excel("cells_metadata.xlsx", index=False)
print("Wrote", os.path.abspath("cells_metadata.xlsx"))
