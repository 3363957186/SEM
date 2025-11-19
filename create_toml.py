import pandas as pd
import numpy as np


def generate_toml_config(csv_file, output_toml="stage_2.toml"):
    """
    根据实际CSV文件自动生成TOML配置文件
    """
    print("正在读取数据...")
    df = pd.read_csv(csv_file)

    print(f"数据维度: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 排除的列
    exclude_columns = ['Label', 'cell_id', 'observation_joinid']

    # 数值型特征和分类特征
    numerical_features = []
    categorical_features = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        # 判断是数值型还是分类型
        if df[col].dtype in ['int64', 'float64']:
            # 检查是否是离散的数值（可能是分类变量）
            unique_count = df[col].nunique()
            if unique_count < 20 and df[col].dtype == 'int64':
                categorical_features.append((col, unique_count))
            else:
                numerical_features.append(col)
        else:
            # 字符串类型，统计类别数
            unique_count = df[col].nunique()
            categorical_features.append((col, unique_count))

    # 生成TOML配置
    with open(output_toml, 'w', encoding='utf-8') as f:
        # Label配置
        f.write("[label]\n")
        f.write("    [label.label]\n")
        f.write("    type = \"categorical\"\n")
        num_labels = df['Label'].nunique()
        f.write(f"    num_categories = {num_labels}  # Braak stage I-VI\n\n")

        # Data配置
        f.write("[data]\n")
        f.write("train_csv = \"train_data_reduced_with_label.csv\"\n")
        f.write("val_csv = \"test_data_reduced_with_label.csv\"\n")

        # 确定ID列
        if 'donor_id' in df.columns:
            f.write("id_column = \"donor_id\"\n\n")
        elif 'cell_id' in df.columns:
            f.write("id_column = \"cell_id\"\n\n")
        else:
            f.write("# id_column = \"your_id_column\"  # 请手动指定\n\n")

        # 数值型特征
        f.write("# ==================== 数值型特征 ====================\n")
        for feat in sorted(numerical_features):
            # 处理特殊字符
            feat_name = feat.replace('.', '_').replace(' ', '_')
            f.write(f'[feature."{feat}"]\n')
            f.write('type = "numerical"\n')
            f.write('shape = [ 1,]\n\n')

        # 分类特征
        f.write("# ==================== 分类特征 ====================\n")
        for feat, num_cats in sorted(categorical_features):
            feat_name = feat.replace('.', '_').replace(' ', '_')
            f.write(f'[feature."{feat}"]\n')
            f.write('type = "categorical"\n')
            f.write(f'num_categories = {num_cats}  # {num_cats} unique values\n\n')

    print(f"\n✓ TOML配置文件已生成: {output_toml}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("配置文件统计")
    print("=" * 60)
    print(f"Label类别数: {num_labels}")
    print(f"数值型特征数: {len(numerical_features)}")
    print(f"分类特征数: {len(categorical_features)}")
    print(f"总特征数: {len(numerical_features) + len(categorical_features)}")

    print("\n【数值型特征列表】")
    for feat in sorted(numerical_features)[:10]:
        print(f"  - {feat}")
    if len(numerical_features) > 10:
        print(f"  ... 还有 {len(numerical_features) - 10} 个特征")

    print("\n【分类特征列表】")
    for feat, num_cats in sorted(categorical_features)[:10]:
        print(f"  - {feat} ({num_cats} 类别)")
    if len(categorical_features) > 10:
        print(f"  ... 还有 {len(categorical_features) - 10} 个特征")

    return numerical_features, categorical_features


if __name__ == "__main__":
    # 使用训练集的删减版生成配置
    csv_file = "train_data_X.csv"

    try:
        numerical_features, categorical_features = generate_toml_config(csv_file)
        print("\n✓ 配置文件生成完成！")
        print("\n【下一步】")
        print("1. 检查生成的 stage_2.toml 文件")
        print("2. 根据需要调整特征类型（数值型 vs 分类型）")
        print("3. 删除不需要的特征")
        print("4. 确认id_column设置正确")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_file}'")
        print("请先运行数据处理脚本生成训练集文件")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback

        traceback.print_exc()