import pandas as pd

# 想保留的前多少列
N_COLS = 100

# 先写一个小函数：读一个 txt → 去掉 ! 行 → 设 index → 转置
def load_and_transpose(txt_path):
    df = pd.read_csv(
        txt_path,
        sep='\t',
        header=0,        # 第一行作为表头（非 ! 开头）
        dtype=str,
        engine='python',
        comment='!'      # 关键：忽略所有以 ! 开头的行
    )

    # 第一列作为行索引（一般是 gene/probe ID）
    df = df.set_index(df.columns[0])

    # 转置：变成每行一个样本
    df_t = df.T.reset_index().rename(columns={"index": "Status"})
    return df_t


# 1. 分别读取并转置 blood1 / blood2
df1_transposed = load_and_transpose('/Users/e/Downloads/blood1.txt')
df2_transposed = load_and_transpose('/Users/e/Downloads/blood2.txt')

# 2. 在“第 5 步之前”把两个表合并（按行拼接：样本叠在一起）
df_merged = pd.concat([df1_transposed, df2_transposed],
                      axis=0,      # 按行拼
                      ignore_index=True)

# 3. 取前 N_COLS 列（包含 Status 在内）
n = min(N_COLS, df_merged.shape[1])
df_filtered = df_merged.iloc[:, :n].copy()   # .copy() 避免 SettingWithCopyWarning

# 4. 把除了 "Status" 以外的列尽量转成数值  ← 你说的“第 5 步”从这里开始
for col in df_filtered.columns:
    if col == 'Status':
        continue
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='ignore')

# 5. 写出合并后的 CSV
df_filtered.to_csv('blood_merged_transposed_cols.csv',
                   index=False,
                   encoding='utf-8-sig')
