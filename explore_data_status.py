"""
数据探索和可视化脚本
用于分析血液数据的STATUS分布，回答Catherine关于MCI分类的问题
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def explore_status_distribution(input_file):
    """
    深入探索Status列的所有可能值及其分布
    回答: 有哪些STATUS? 它们的分布如何? MCI应该归为哪一类?
    """
    print("\n" + "=" * 70)
    print("STATUS 分布详细分析")
    print("=" * 70)

    df = pd.read_csv(input_file)
    print(f"\n总样本数: {len(df)}")

    # 1. 打印所有唯一的STATUS值
    print("\n" + "-" * 70)
    print("所有唯一的STATUS值:")
    print("-" * 70)
    unique_statuses = sorted(df['Status'].unique())

    status_data = []
    for status in unique_statuses:
        count = (df['Status'] == status).sum()
        percentage = count / len(df) * 100
        status_data.append({
            'Status': status,
            'Count': count,
            'Percentage': percentage
        })
        print(f"  {status:<30} {count:>6} ({percentage:>5.1f}%)")

    # 2. 按主要类别分组
    print("\n" + "-" * 70)
    print("按主要疾病类别分组:")
    print("-" * 70)

    def categorize_status(status):
        """将Status归类到主要类别"""
        status = str(status).upper()
        if 'CTL' in status or 'CONTROL' in status:
            return 'CTL'
        elif 'MCI' in status:
            return 'MCI'
        elif 'AD' in status:
            return 'AD'
        else:
            return 'OTHER'

    df['Status_Category'] = df['Status'].apply(categorize_status)

    category_counts = df['Status_Category'].value_counts().sort_index()

    for category in ['CTL', 'MCI', 'AD', 'OTHER']:
        if category in category_counts.index:
            count = category_counts[category]
            percentage = count / len(df) * 100
            print(f"  {category:<10} {count:>6} ({percentage:>5.1f}%)")

    # 3. 检查类别平衡性
    print("\n" + "-" * 70)
    print("类别平衡性分析:")
    print("-" * 70)

    if 'CTL' in category_counts.index and 'MCI' in category_counts.index:
        ctl_count = category_counts['CTL']
        mci_count = category_counts['MCI']
        ratio = ctl_count / mci_count if mci_count > 0 else float('inf')
        print(f"  CTL vs MCI 比例: {ratio:.3f}")
        if 0.8 <= ratio <= 1.2:
            print("  ✓ CTL和MCI相对平衡")
        else:
            print(f"  ⚠ CTL和MCI不平衡 (比例 {ratio:.2f})")

    if 'CTL' in category_counts.index and 'AD' in category_counts.index:
        ctl_count = category_counts['CTL']
        ad_count = category_counts['AD']
        ratio = ctl_count / ad_count if ad_count > 0 else float('inf')
        print(f"  CTL vs AD 比例: {ratio:.3f}")

    if 'MCI' in category_counts.index and 'AD' in category_counts.index:
        mci_count = category_counts['MCI']
        ad_count = category_counts['AD']
        ratio = mci_count / ad_count if ad_count > 0 else float('inf')
        print(f"  MCI vs AD 比例: {ratio:.3f}")

    # 4. 当前标签策略分析
    print("\n" + "-" * 70)
    print("当前标签策略分析 (包含'AD'的为1, 其他为0):")
    print("-" * 70)

    current_labels = df['Status'].str.contains('AD', case=False, na=False).astype(int)
    print(f"  Label 1 (包含'AD'的): {current_labels.sum()} ({current_labels.sum() / len(df) * 100:.1f}%)")
    print(f"  Label 0 (不包含'AD'的): {(1 - current_labels).sum()} ({(1 - current_labels).sum() / len(df) * 100:.1f}%)")

    # 显示每个类别的标签分布
    print("\n  各类别的标签分配:")
    for category in ['CTL', 'MCI', 'AD']:
        if category in df['Status_Category'].unique():
            mask = df['Status_Category'] == category
            label_1_count = current_labels[mask].sum()
            label_0_count = (~current_labels.astype(bool))[mask].sum()
            print(f"    {category}: Label_1={label_1_count}, Label_0={label_0_count}")

    return df, category_counts


def plot_status_distribution(df, category_counts, output_dir="./figures"):
    """
    绘制所有STATUS类别的分布图
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("生成可视化图表...")
    print("=" * 70)

    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11

    # 1. 条形图 - 主要类别分布
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = []
    counts = []
    colors_map = {'CTL': '#2ecc71', 'MCI': '#f39c12', 'AD': '#e74c3c', 'OTHER': '#95a5a6'}
    colors = []

    for cat in ['CTL', 'MCI', 'AD', 'OTHER']:
        if cat in category_counts.index:
            categories.append(cat)
            counts.append(category_counts[cat])
            colors.append(colors_map[cat])

    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = count / len(df) * 100
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_xlabel('Disease Status Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Disease Status Categories', fontsize=15, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/status_distribution_bar.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_dir}/status_distribution_bar.png")
    plt.close()

    # 2. 饼图
    fig, ax = plt.subplots(figsize=(9, 9))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=categories,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        explode=[0.05] * len(categories)
    )

    # 添加样本数
    for i, (text, autotext, count) in enumerate(zip(texts, autotexts, counts)):
        autotext.set_color('white')
        autotext.set_fontsize(12)

    ax.set_title('Status Distribution (Pie Chart)', fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/status_distribution_pie.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_dir}/status_distribution_pie.png")
    plt.close()

    # 3. CTL vs MCI 详细对比（Catherine特别提到的）
    if 'CTL' in category_counts.index and 'MCI' in category_counts.index:
        fig, ax = plt.subplots(figsize=(8, 6))

        ctl_mci_categories = ['CTL', 'MCI']
        ctl_mci_counts = [category_counts['CTL'], category_counts['MCI']]
        ctl_mci_colors = [colors_map['CTL'], colors_map['MCI']]

        bars = ax.bar(ctl_mci_categories, ctl_mci_counts, color=ctl_mci_colors,
                      edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, count in zip(bars, ctl_mci_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{count}',
                    ha='center', va='bottom', fontweight='bold', fontsize=14)

        ax.set_xlabel('Status', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
        ax.set_title('CTL vs MCI Distribution\n(As mentioned by Catherine)',
                     fontsize=15, fontweight='bold')

        # 添加平衡性注释
        ratio = ctl_mci_counts[0] / ctl_mci_counts[1]
        balance_text = "Balanced" if 0.8 <= ratio <= 1.2 else f"Imbalanced (ratio: {ratio:.2f})"
        ax.text(0.5, 0.95, balance_text,
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow' if 0.8 <= ratio <= 1.2 else 'orange', alpha=0.7),
                fontsize=12, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/ctl_vs_mci.png', dpi=300, bbox_inches='tight')
        print(f"✓ 保存: {output_dir}/ctl_vs_mci.png")
        plt.close()

    # 4. 不同标签策略的对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 策略1: 当前策略 (AD包含的为1)
    ax = axes[0]
    current_labels = df['Status'].str.contains('AD', case=False, na=False).astype(int)
    current_counts = pd.Series(current_labels).value_counts().sort_index()

    bars = ax.bar(['Non-AD (0)', 'AD (1)'],
                  [current_counts.get(0, 0), current_counts.get(1, 0)],
                  color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
    ax.set_title('Strategy 1: Current\n(MCI → Non-AD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # 策略2: MCI归为AD
    ax = axes[1]

    def label_strategy_2(status):
        status = str(status).upper()
        return 1 if ('AD' in status or 'MCI' in status) else 0

    labels_2 = df['Status'].apply(label_strategy_2)
    counts_2 = pd.Series(labels_2).value_counts().sort_index()

    bars = ax.bar(['Non-AD (0)', 'AD+MCI (1)'],
                  [counts_2.get(0, 0), counts_2.get(1, 0)],
                  color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.5)
    ax.set_title('Strategy 2: Option B\n(MCI → AD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # 策略3: 三分类
    ax = axes[2]
    category_mapping = {'CTL': 0, 'MCI': 1, 'AD': 2, 'OTHER': 3}
    labels_3 = df['Status_Category'].map(category_mapping)
    counts_3 = pd.Series(labels_3).value_counts().sort_index()

    labels_list = []
    counts_list = []
    colors_list = []
    for i, (label, color) in enumerate(zip(['CTL (0)', 'MCI (1)', 'AD (2)', 'OTHER (3)'],
                                           ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6'])):
        if i in counts_3.index:
            labels_list.append(label)
            counts_list.append(counts_3[i])
            colors_list.append(color)

    bars = ax.bar(labels_list, counts_list, color=colors_list,
                  edgecolor='black', linewidth=1.5)
    ax.set_title('Strategy 3: Multi-class\n(CTL/MCI/AD separate)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/labeling_strategies_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {output_dir}/labeling_strategies_comparison.png")
    plt.close()

    print("\n所有图表已生成！")


def generate_recommendation_report(df, category_counts, output_dir="./reports"):
    """
    生成推荐报告，帮助决定MCI应该归为哪一类
    """
    os.makedirs(output_dir, exist_ok=True)

    report_path = f'{output_dir}/mci_classification_recommendation.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MCI分类策略推荐报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("背景:\n")
        f.write("-" * 70 + "\n")
        f.write("Catherine指出需要确认Foundation Model是基于什么数据训练的，\n")
        f.write("以决定MCI(轻度认知障碍)应该归为AD还是Non-AD。\n\n")

        f.write("数据集概况:\n")
        f.write("-" * 70 + "\n")
        f.write(f"总样本数: {len(df)}\n\n")

        for cat in ['CTL', 'MCI', 'AD']:
            if cat in category_counts.index:
                count = category_counts[cat]
                percentage = count / len(df) * 100
                f.write(f"{cat:>10}: {count:>6} ({percentage:>5.1f}%)\n")

        f.write("\n")
        f.write("三种可能的标签策略:\n")
        f.write("=" * 70 + "\n\n")

        # 策略1
        f.write("策略1: MCI归为Non-AD (当前策略)\n")
        f.write("-" * 70 + "\n")
        f.write("适用条件: Foundation Model只在AD晚期数据上训练\n")
        current_labels = df['Status'].str.contains('AD', case=False, na=False).astype(int)
        label_1 = current_labels.sum()
        label_0 = len(df) - label_1
        f.write(f"  Label 0 (Non-AD, 包括CTL+MCI): {label_0} ({label_0 / len(df) * 100:.1f}%)\n")
        f.write(f"  Label 1 (AD): {label_1} ({label_1 / len(df) * 100:.1f}%)\n")
        f.write(f"  类别比例 (AD/Non-AD): {label_1 / label_0:.3f}\n")
        f.write(f"  优点: 如果Foundation Model在AD晚期训练，这样分类更合理\n")
        f.write(f"  缺点: 将MCI(AD早期)和CTL(健康对照)混在一起可能丢失信息\n\n")

        # 策略2
        f.write("策略2: MCI归为AD (推荐用于早期诊断)\n")
        f.write("-" * 70 + "\n")
        f.write("适用条件: Foundation Model在AD所有阶段(包括早期/MCI)训练\n")

        def label_strategy_2(status):
            status = str(status).upper()
            return 1 if ('AD' in status or 'MCI' in status) else 0

        labels_2 = df['Status'].apply(label_strategy_2)
        label_1 = labels_2.sum()
        label_0 = len(df) - label_1
        f.write(f"  Label 0 (Non-AD, 只有CTL): {label_0} ({label_0 / len(df) * 100:.1f}%)\n")
        f.write(f"  Label 1 (AD+MCI): {label_1} ({label_1 / len(df) * 100:.1f}%)\n")
        f.write(f"  类别比例 (AD+MCI/CTL): {label_1 / label_0:.3f}\n")
        f.write(f"  优点: 更符合疾病连续性，有助于早期诊断\n")
        f.write(f"  缺点: 如果Foundation Model不包含MCI，可能不匹配\n\n")

        # 策略3
        f.write("策略3: 三分类模型\n")
        f.write("-" * 70 + "\n")
        f.write("适用条件: 需要区分不同疾病阶段\n")
        f.write(f"  Label 0 (CTL): {category_counts.get('CTL', 0)}\n")
        f.write(f"  Label 1 (MCI): {category_counts.get('MCI', 0)}\n")
        f.write(f"  Label 2 (AD): {category_counts.get('AD', 0)}\n")
        f.write(f"  优点: 保留了完整的疾病阶段信息\n")
        f.write(f"  缺点: 需要更多数据，模型更复杂\n\n")

        f.write("\n")
        f.write("推荐行动:\n")
        f.write("=" * 70 + "\n")
        f.write("1. 确认Foundation Model的训练数据:\n")
        f.write("   - 查看模型文档或论文\n")
        f.write("   - 联系模型开发团队\n")
        f.write("   - 检查模型的标签定义\n\n")

        f.write("2. 根据确认结果选择策略:\n")
        f.write("   IF Foundation Model训练于AD晚期:\n")
        f.write("      → 使用策略1 (MCI归为Non-AD)\n")
        f.write("   ELSE IF Foundation Model训练于AD所有阶段:\n")
        f.write("      → 使用策略2 (MCI归为AD) ← 推荐用于早期诊断任务\n")
        f.write("   ELSE IF 不确定或需要更精细的分类:\n")
        f.write("      → 考虑策略3 (三分类)\n\n")

        f.write("3. 验证选择:\n")
        f.write("   - 在小规模数据上测试不同策略\n")
        f.write("   - 比较验证集性能\n")
        f.write("   - 咨询领域专家意见\n\n")

        f.write("=" * 70 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 70 + "\n")

    print(f"\n✓ 推荐报告已生成: {report_path}")

    # 同时在控制台打印关键信息
    print("\n" + "=" * 70)
    print("关键决策问题:")
    print("=" * 70)
    print("\n您需要向Catherine或查阅Foundation Model文档确认:")
    print("Foundation Model是基于什么数据训练的?")
    print("  [ ] 只有AD晚期 → 选择策略1 (MCI归为Non-AD)")
    print("  [ ] AD所有阶段(包括早期/MCI) → 选择策略2 (MCI归为AD)")
    print("  [ ] 需要三分类 → 选择策略3")
    print("\n确认后请修改 blood_data.py 中的标签生成逻辑")
    print("=" * 70)


if __name__ == "__main__":
    # 配置
    input_file = "blood_merged_transposed_cols.csv"

    print("\n" + "=" * 70)
    print(" 血液数据STATUS分布分析")
    print(" 回答Catherine的问题: MCI应该归为哪一类?")
    print("=" * 70)

    try:
        # 1. 探索数据分布
        df, category_counts = explore_status_distribution(input_file)

        # 2. 生成可视化图表
        plot_status_distribution(df, category_counts)

        # 3. 生成推荐报告
        generate_recommendation_report(df, category_counts)

        print("\n" + "=" * 70)
        print("✅ 分析完成！")
        print("=" * 70)
        print("\n生成的文件:")
        print("  - ./figures/status_distribution_bar.png")
        print("  - ./figures/status_distribution_pie.png")
        print("  - ./figures/ctl_vs_mci.png")
        print("  - ./figures/labeling_strategies_comparison.png")
        print("  - ./reports/mci_classification_recommendation.txt")

    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 '{input_file}'")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback

        traceback.print_exc()