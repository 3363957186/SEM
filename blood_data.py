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


def convert_label_to_onehot(df, num_classes):
    """
    将单列Label转换为one-hot编码的多列

    Parameters:
    -----------
    df : DataFrame
        包含Label列的数据框
    num_classes : int
        类别数量

    Returns:
    --------
    df_onehot : DataFrame
        Label列替换为Label_0, Label_1的数据框
    """

    print(f"\n将Label转换为one-hot编码（{num_classes}个类别）...")

    # 提取Label值
    label_values = df['Label'].values.astype(int)

    # 创建one-hot编码
    one_hot = np.zeros((len(df), num_classes))
    one_hot[np.arange(len(df)), label_values] = 1

    # 复制数据框并删除原始Label列
    df_onehot = df.drop(columns=['Label']).copy()

    # 添加one-hot编码列
    for i in range(num_classes):
        df_onehot[f'Label_{i}'] = one_hot[:, i].astype(int)

    print(f"✓ Label列已转换:")
    print(f"  原始: Label (单列，值为0-{num_classes - 1})")
    print(f"  转换后: Label_0, Label_1, ..., Label_{num_classes - 1} (one-hot编码)")

    # 显示示例
    print(f"\n示例（前3行）:")
    print(f"  原始Label: {label_values[:3]}")
    for i in range(num_classes):
        print(f"  Label_{i}: {one_hot[:3, i]}")

    return df_onehot


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


def create_label_strategy_2(status):
    status = str(status).upper()
    return 1 if ('AD' in status or 'MCI' in status) else 0


def process_blood_data_with_split(input_file, test_size=0.2, random_state=42):
    """
    处理血液表达数据，创建Label列并分割为训练集和测试集

    Label创建规则：
    - Status列包含"AD"（如"status: AD"、"status: AD.1"等）→ Label = 1
    - 其他情况 → Label = 0

    功能：
    - 对所有分类特征进行标签编码
    - 将Label转换为one-hot编码
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

    df['Label'] = df['Status'].apply(create_label_strategy_2)

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

    # ============= 将Label转换为one-hot编码 =============
    print("\n" + "=" * 70)
    print("步骤7: 将Label转换为one-hot编码")
    print("=" * 70)

    train_onehot = convert_label_to_onehot(train_encoded, num_label_classes)
    test_onehot = convert_label_to_onehot(test_encoded, num_label_classes)

    print("\n" + "=" * 70)
    print("步骤8: 保存文件")
    print("=" * 70)

    # 保存one-hot编码后的文件（用于训练）
    output_train_encoded = "data/train_data_blood.csv"
    train_onehot.to_csv(output_train_encoded, index=False)
    print(f"✓ 训练集（one-hot编码）: {output_train_encoded}")
    print(f"    维度: {train_onehot.shape}")
    print(f"    列: 前10列 = {train_onehot.columns.tolist()[:10]}")

    output_test_encoded = "data/test_data_blood.csv"
    test_onehot.to_csv(output_test_encoded, index=False)
    print(f"\n✓ 测试集（one-hot编码）: {output_test_encoded}")
    print(f"    维度: {test_onehot.shape}")

    # 保存原始Label版本（未one-hot，用于分析）
    output_train_original = "data/train_data_blood_original.csv"
    train_encoded.to_csv(output_train_original, index=False)
    print(f"\n✓ 训练集（原始Label）: {output_train_original}")

    output_test_original = "data/test_data_blood_original.csv"
    test_encoded.to_csv(output_test_original, index=False)
    print(f"✓ 测试集（原始Label）: {output_test_original}")

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

    generate_toml_config(train_onehot, encoders, categorical_cols, num_label_classes, "blood_model.toml")

    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)

    print("\n【数据统计】")
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"测试集: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"\n特征统计:")
    num_features_without_label = train_onehot.shape[1] - num_label_classes
    print(f"  探针特征 (ILMN_*): {len([c for c in train_encoded.columns if c.startswith('ILMN_') and c != 'Label'])}")
    print(f"  分类特征（已编码）: {len(categorical_cols)}")
    print(f"  总特征数: {num_features_without_label}")
    print(f"  Label列数（one-hot）: {num_label_classes}")

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
    print(f"1. {output_train_encoded} - 训练集（清理+编码，用于训练）⭐")
    print(f"2. {output_test_encoded} - 测试集（清理+编码，用于测试）⭐")
    print(f"3. {output_train_original} - 训练集（清理+原始Label）")
    print(f"4. {output_test_original} - 测试集（清理+原始Label）")
    print(f"5. {output_train_full} - 训练集完整版（调试用）")
    print(f"6. {output_test_full} - 测试集完整版（调试用）")
    print(f"7. blood_model.toml - 模型配置文件 ⭐")
    if encoders:
        print(f"8. data/categorical_encoders.json - 分类特征编码映射")

    print("\n【下一步】")
    print("1. 检查 train_data_blood.csv 确认列已正确删除")
    print("2. 使用训练集进行模型训练")

    return train_onehot, test_onehot, encoders, num_label_classes


def generate_toml_config(df, encoders, categorical_cols, num_label_classes, output_toml):
    """
    根据one-hot编码后的数据框生成TOML配置文件
    """
    # Label列现在是Label_0, Label_1
    label_columns = [f'Label_{i}' for i in range(num_label_classes)]

    numerical_features = []
    categorical_features = []

    for col in df.columns:
        if col in label_columns:
            continue

        if col in categorical_cols:
            # 使用编码器中保存的类别数
            num_cats = encoders[col]['num_categories']
            categorical_features.append((col, num_cats))
        else:
            numerical_features.append(col)

    with open(output_toml, 'w', encoding='utf-8') as f:
        f.write("# Blood Expression Data - AD Classification Model Configuration\n")
        f.write("# Auto-generated with one-hot encoded labels\n")
        f.write("# Label: 1 = AD (Alzheimer's Disease), 0 = Non-AD\n\n")

        # Label配置 - 每个one-hot列都是一个二分类
        f.write("[label]\n")
        for i in range(num_label_classes):
            f.write(f"    [label.Label_{i}]\n")
            f.write("    type = \"categorical\"\n")
            f.write(f"    num_categories = 2  # Binary: 0 or 1\n")
            if i < num_label_classes - 1:
                f.write("\n")
        f.write("\n")

        # Data配置
        f.write("[data]\n")
        f.write("train_csv = \"data/train_data_blood.csv\"\n")
        f.write("val_csv = \"data/test_data_blood.csv\"\n")
        f.write("# Each row is an independent blood sample\n\n")

        # 数值型特征（探针表达值）
        f.write(f"# ==================== Numerical Features - Probe Expression ({len(numerical_features)}) ====================\n")
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
    print(f"  - Label: {num_label_classes}个one-hot列 (Label_0 到 Label_{num_label_classes - 1})")
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