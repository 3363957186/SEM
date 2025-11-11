# python split_by_ad_from_disease.py
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

SRC = "data/donor_features.csv"   # 你的CSV
OUTDIR = Path("pseudodata")
OUTDIR.mkdir(parents=True, exist_ok=True)

def ensure_donor_id(df: pd.DataFrame) -> pd.DataFrame:
    # donor_id 可能在索引或列里，这里统一成显式列
    if "donor_id" not in df.columns:
        if df.index.name == "donor_id":
            df = df.reset_index()
        else:
            # 没名字的索引，也当成 donor_id
            df = df.reset_index().rename(columns={"index": "donor_id"})
    return df

# 定义“是否AD”的判定：只要包含 alzheimer 关键字（大小写不敏感）
AD_PATTERN = re.compile(r"alzheimer'?s?", re.IGNORECASE)

def has_ad(val) -> bool:
    if pd.isna(val):
        return False
    s = str(val)
    return AD_PATTERN.search(s) is not None

print(f"Loading {SRC} ...")
df = pd.read_csv(SRC)
df = ensure_donor_id(df)

assert "disease" in df.columns, "找不到 disease 列，请确认 donor_features.csv 的列名。"

# 生成二分类标签
df["label"] = df["disease"].apply(has_ad).astype(int)

# 丢掉没有 donor_id 的行
df = df[df["donor_id"].notna()].copy()
df["donor_id"] = df["donor_id"].astype(str)

# 看一下类比例
counts = df["label"].value_counts(dropna=False).to_dict()
print("Label distribution:", counts)

# 若某一类太少，给出提示
if min(counts.get(0,0), counts.get(1,0)) < 10:
    print("[WARN] 某一类样本 < 10，分层切分可能不稳定，请检查 disease 标注或放宽判定规则。")

# 分层切分 70/15/15
train_df, temp = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp, test_size=0.50, stratify=temp["label"], random_state=42
)

# 保存：含 label 的版本（便于检查）；以及去泄漏的特征版（去掉 disease/label）
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    part.to_csv(OUTDIR / f"{name}.csv", index=False)

    X = part.drop(columns=[c for c in ["disease"] if c in part.columns])
    X.to_csv(OUTDIR / f"{name}_X.csv", index=False)

print("Saved:",
      OUTDIR / "train.csv", OUTDIR / "val.csv", OUTDIR / "test.csv",
      OUTDIR / "train_X.csv", OUTDIR / "val_X.csv", OUTDIR / "test_X.csv")

# 额外输出一个供你核对的对照表：每类各取前20条
pd.concat([
    train_df.query("label==1").head(20)[["donor_id","disease","label"]],
    train_df.query("label==0").head(20)[["donor_id","disease","label"]],
]).to_csv(OUTDIR / "label_sanity_check.csv", index=False)
print("Saved:", OUTDIR / "label_sanity_check.csv")
