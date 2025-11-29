import pandas as pd
import numpy as np
import os
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_column_name(col):
    """
    清理列名，将特殊字符替换为下划线
    """
    if col == 'Label' or col.startswith('Label_'):
        return col

    col = re.sub(r'[.\s/\-\(\):]', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')

    return col


def encode_categorical_features(train_df, test_df, exclude_columns):
    """
    对分类特征进行标签编码

    Parameters:
    -----------
    train_df : DataFrame
        训练集
    test_df : DataFrame
        测试集
    exclude_columns : list
        不需要编码的列

    Returns:
    --------
    train_encoded, test_encoded, encoders_dict, categorical_columns
    """
    print("\n" + "=" * 70)
    print("步骤: 对分类特征进行编码")
    print("=" * 70)

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    encoders = {}
    categorical_columns = []

    for col in train_df.columns:
        if col in exclude_columns:
            continue

        # 检查是否是字符串类型（分类变量）
        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
            categorical_columns.append(col)

            print(f"\n编码列: {col}")
            print(f"  唯一值数量: {train_df[col].nunique()}")

            # 创建标签编码器
            le = LabelEncoder()

            # 合并训练集和测试集的所有类别
            all_categories = pd.concat([train_df[col], test_df[col]]).unique()
            le.fit(all_categories)

            # 编码
            train_encoded[col] = le.transform(train_df[col])
            test_encoded[col] = le.transform(test_df[col])

            # 保存编码器
            encoders[col] = {
                'encoder': le,
                'classes': le.classes_.tolist(),
                'num_categories': len(le.classes_)
            }

            print(f"  编码范围: 0 - {len(le.classes_) - 1}")
            print(f"  示例: {train_df[col].iloc[0]} -> {train_encoded[col].iloc[0]}")

    if not categorical_columns:
        print("\n✓ 没有需要编码的分类特征")
    else:
        print(f"\n✓ 共编码了 {len(categorical_columns)} 个分类特征")

    return train_encoded, test_encoded, encoders, categorical_columns


def save_encoders(encoders, filepath):
    """
    保存编码器信息到JSON文件
    """
    encoders_info = {}
    for col, info in encoders.items():
        encoders_info[col] = {
            'classes': info['classes'],
            'num_categories': info['num_categories']
        }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(encoders_info, f, indent=2, ensure_ascii=False)

    print(f"✓ 编码器信息已保存: {filepath}")


def process_blood_data_with_split(input_file, test_size=0.2, random_state=42):
    """
    处理血液表达数据，创建Label列并分割为训练集和测试集

    Label创建规则：
    - Status列包含"AD"（如"status: AD"、"status: AD.1"等）→ Label = 1
    - 其他情况 → Label = 0

    功能：
    - 对所有分类特征进行标签编码
    - 保留单个Label列（二分类：0/1）
    - 删除可能导致数据泄露的列
    - 保存编码器映射
    """

    # 创建输出目录
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("步骤1: 读取数据")
    print("=" * 70)
    df = pd.read_csv(input_file)

    print(f"✓ 数据维度: {df.shape}")
    print(f"✓ 样本数: {len(df)}")
    print(f"✓ 特征数: {len(df.columns)}")

    print("\n" + "=" * 70)
    print("步骤2: 清理列名（去除特殊字符）")
    print("=" * 70)

    column_mapping = {}
    changed_columns = []

    for old_col in df.columns:
        new_col = clean_column_name(old_col)
        column_mapping[old_col] = new_col
        if old_col != new_col:
            changed_columns.append((old_col, new_col))
            print(f"  {old_col} → {new_col}")

    if not changed_columns:
        print("  ✓ 没有需要修改的列名")
    else:
        print(f"\n  ✓ 修改了 {len(changed_columns)} 个列名")

    df.columns = [column_mapping[col] for col in df.columns]

    print("\n" + "=" * 70)
    print("步骤3: 创建Label列")
    print("=" * 70)

    if 'Status' not in df.columns:
        print("❌ 错误：找不到Status列！")
        print(f"可用列名: {df.columns.tolist()}")
        return None

    # 创建Label: Status包含"AD"为1，否则为0
    df['Label'] = df['Status'].str.contains('AD', case=False, na=False).astype(int)

    print(f"Status唯一值示例: {sorted(df['Status'].unique())[:10]}")
    print(f"\nLabel分布:")
    print(df['Label'].value_counts().sort_index())
    print(f"\n各类别样本数:")
    for label in sorted(df['Label'].unique()):
        count = (df['Label'] == label).sum()
        percentage = count / len(df) * 100
        label_name = "AD (Alzheimer's Disease)" if label == 1 else "Non-AD (Control/MCI/Other)"
        print(f"  Label {label} ({label_name}): {count} ({percentage:.1f}%)")

    # 确定类别数（二分类：0和1）
    num_label_classes = int(df['Label'].nunique())
    print(f"\n✓ Label类别数: {num_label_classes}")

    print("\n" + "=" * 70)
    print("步骤4: 分割训练集和测试集")
    print("=" * 70)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Label']
    )

    print(f"✓ 训练集维度: {train_df.shape}")
    print(f"✓ 测试集维度: {test_df.shape}")

    print("\n" + "=" * 70)
    print("步骤5: 删除可能泄露的列")
    print("=" * 70)

    # 要删除的列
    columns_to_remove = [
        'Status',  # 直接关联诊断
        'ID_REF',  # 样本ID可能编码批次或患者信息
    ]

    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]

    print(f"将要删除的列 ({len(existing_columns_to_remove)}):")
    for col in existing_columns_to_remove:
        print(f"  - {col}")

    train_df_reduced = train_df.drop(columns=existing_columns_to_remove, errors='ignore')
    test_df_reduced = test_df.drop(columns=existing_columns_to_remove, errors='ignore')

    print(f"\n✓ 删除后训练集维度: {train_df_reduced.shape}")
    print(f"✓ 删除后测试集维度: {test_df_reduced.shape}")

    # 显示保留的列
    print(f"\n保留的列 ({len(train_df_reduced.columns) - 1}): ")  # -1 for Label
    retained_cols = [c for c in train_df_reduced.columns if c != 'Label']
    print(f"  - {len([c for c in retained_cols if c.startswith('ILMN_')])} 个探针特征 (ILMN_*)")
    print(f"  - {len([c for c in retained_cols if not c.startswith('ILMN_')])} 个其他特征")

    # ============= 编码分类特征 =============
    print("\n" + "=" * 70)
    print("步骤6: 编码分类特征")
    print("=" * 70)

    exclude_for_encoding = ['Label']

    train_encoded, test_encoded, encoders, categorical_cols = encode_categorical_features(
        train_df_reduced,
        test_df_reduced,
        exclude_for_encoding
    )

    # 保存编码器信息
    if encoders:
        save_encoders(encoders, "data/categorical_encoders.json")

    # ============= 二分类直接使用Label列，无需one-hot =============
    print("\n" + "=" * 70)
    print("步骤7: 保留Label列（二分类，无需one-hot编码）")
    print("=" * 70)

    print(f"✓ Label列为二分类（0/1），直接用于训练")
    print(f"  Label 0 (Non-AD): {(train_encoded['Label'] == 0).sum()} 样本")
    print(f"  Label 1 (AD): {(train_encoded['Label'] == 1).sum()} 样本")

    train_final = train_encoded
    test_final = test_encoded

    print("\n" + "=" * 70)
    print("步骤8: 保存文件")
    print("=" * 70)

    # 保存训练集和测试集
    output_train_encoded = "data/train_data_blood.csv"
    train_final.to_csv(output_train_encoded, index=False)
    print(f"✓ 训练集: {output_train_encoded}")
    print(f"    维度: {train_final.shape}")
    print(f"    列: 前10列 = {train_final.columns.tolist()[:10]}")

    output_test_encoded = "data/test_data_blood.csv"
    test_final.to_csv(output_test_encoded, index=False)
    print(f"\n✓ 测试集: {output_test_encoded}")
    print(f"    维度: {test_final.shape}")

    # 保存完整版（包含删除的列，用于调试）
    output_train_full = "data/train_data_blood_full.csv"
    train_df.to_csv(output_train_full, index=False)
    print(f"\n✓ 训练集（完整版，包含所有列）: {output_train_full}")

    output_test_full = "data/test_data_blood_full.csv"
    test_df.to_csv(output_test_full, index=False)
    print(f"✓ 测试集（完整版，包含所有列）: {output_test_full}")

    print("\n" + "=" * 70)
    print("步骤9: 生成TOML配置文件")
    print("=" * 70)

    generate_toml_config(train_final, encoders, categorical_cols, "blood_model.toml")

    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)

    print("\n【数据统计】")
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"测试集: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"\n特征统计:")
    num_features_without_label = train_final.shape[1] - 1  # -1 for Label
    print(f"  探针特征 (ILMN_*): {len([c for c in train_final.columns if c.startswith('ILMN_') and c != 'Label'])}")
    print(f"  分类特征（已编码）: {len(categorical_cols)}")
    print(f"  总特征数: {num_features_without_label}")
    print(f"  Label列: 1 (二分类: 0/1)")

    print("\n【Label分布】")
    print(f"{'Label':<10} {'训练集':<15} {'测试集':<15} {'总计':<15}")
    print("-" * 60)
    for label in sorted(df['Label'].unique()):
        train_count = (train_encoded['Label'] == label).sum()
        test_count = (test_encoded['Label'] == label).sum()
        total_count = (df['Label'] == label).sum()
        label_name = "AD" if label == 1 else "Non-AD"
        print(f"{label_name:<10} {train_count:<15} {test_count:<15} {total_count:<15}")

    print("\n【生成的文件】")
    print(f"1. {output_train_encoded} - 训练集（用于训练）⭐")
    print(f"2. {output_test_encoded} - 测试集（用于测试）⭐")
    print(f"3. {output_train_full} - 训练集完整版（调试用）")
    print(f"4. {output_test_full} - 测试集完整版（调试用）")
    print(f"5. blood_model.toml - 模型配置文件 ⭐")
    if encoders:
        print(f"6. data/categorical_encoders.json - 分类特征编码映射")

    print("\n【下一步】")
    print("1. 检查 train_data_blood.csv 确认列已正确删除")
    print("2. 使用训练集进行模型训练")

    return train_final, test_final, encoders


def generate_toml_config(df, encoders, categorical_cols, output_toml):
    """
    根据数据框生成TOML配置文件（二分类，单个Label列）
    """
    numerical_features = []
    categorical_features = []

    for col in df.columns:
        if col == 'Label':
            continue

        if col in categorical_cols:
            # 使用编码器中保存的类别数
            num_cats = encoders[col]['num_categories']
            categorical_features.append((col, num_cats))
        else:
            numerical_features.append(col)

    with open(output_toml, 'w', encoding='utf-8') as f:
        f.write("# Blood Expression Data - AD Classification Model Configuration\n")
        f.write("# Binary Classification: Label 1 = AD, Label 0 = Non-AD\n\n")

        # Label配置 - 二分类
        f.write("[label]\n")
        f.write("    [label.Label]\n")
        f.write("    type = \"categorical\"\n")
        f.write("    num_categories = 2  # Binary: 0 (Non-AD) or 1 (AD)\n\n")

        # Data配置
        f.write("[data]\n")
        f.write("train_csv = \"data/train_data_blood.csv\"\n")
        f.write("val_csv = \"data/test_data_blood.csv\"\n")
        f.write("# Each row is an independent blood sample\n\n")

        # 数值型特征（探针表达值）
        f.write(
            f"# ==================== Numerical Features - Probe Expression ({len(numerical_features)}) ====================\n")
        for feat in sorted(numerical_features):
            f.write(f'[feature.{feat}]\n')
            f.write('type = "numerical"\n')
            f.write('shape = [ 1,]\n\n')

        # 分类特征（已编码）
        if categorical_features:
            f.write(
                f"# ==================== Categorical Features (Encoded) ({len(categorical_features)}) ====================\n")
            for feat, num_cats in sorted(categorical_features):
                f.write(f'[feature.{feat}]  # Encoded: 0-{num_cats - 1}\n')
                f.write('type = "categorical"\n')
                f.write(f'num_categories = {num_cats}\n\n')

    print(f"✓ TOML配置已生成: {output_toml}")
    print(f"  - Label: 单列二分类 (0=Non-AD, 1=AD)")
    print(f"  - 探针表达特征: {len(numerical_features)}")
    print(f"  - 分类特征: {len(categorical_features)}")


if __name__ == "__main__":
    input_file = "blood_merged_transposed_cols.csv"

    print("\n" + "=" * 70)
    print(" 血液表达数据处理脚本 - AD分类")
    print("=" * 70)
    print(f"\n输入文件: {input_file}\n")

    try:
        result = process_blood_data_with_split(
            input_file,
            test_size=0.2,
            random_state=42
        )

        if result is not None:
            print("\n" + "=" * 70)
            print("✅ 所有文件创建成功！")
            print("=" * 70)

    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback

        traceback.print_exc()