#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import numpy as np

# 扫描NIfTI文件
nifti_dir = Path("data/nacc_nifti")
nifti_files = list(nifti_dir.glob("*.nii.gz")) + list(nifti_dir.glob("*.nii"))

print(f"找到 {len(nifti_files)} 个NIfTI文件")

# 创建数据框
data = []
for nifti_file in nifti_files:
    patient_id = nifti_file.stem.replace('.nii', '')
    
    # 创建一条记录
    record = {
        'patient_id': patient_id,
        'MRI_path': str(nifti_file.absolute()),  # 绝对路径
        'MRI_path_list': [str(nifti_file.absolute())],  # 列表形式（代码期望的格式）
        # 以下是占位符，实际应该从NACC临床数据获取
        'NC': 0,  # Normal Control
        'MCI': 0,  # Mild Cognitive Impairment  
        'AD': 1,  # Alzheimer's Disease (占位符)
        'age': np.nan,
        'sex': np.nan,
    }
    data.append(record)

df = pd.DataFrame(data)

# 保存CSV
output_csv = "data/nacc_mri_data.csv"
df.to_csv(output_csv, index=False)
print(f"\n✓ CSV已保存: {output_csv}")
print(f"\n前5行预览:")
print(df.head())
print(f"\nCSV列: {df.columns.tolist()}")

