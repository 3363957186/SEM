#!/usr/bin/env python3
"""
train_catboost_multilabel.py - CatBoost多标签训练

支持Label_0, Label_1, Label_2, Label_3等多个标签列
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CatBoost Multi-Label Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据路径
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--vld_path', type=str, required=True)

    # 标签选择
    parser.add_argument('--label_col', type=str, default='Label_0',
                        help='Which label column to use (Label_0, Label_1, etc.)')
    parser.add_argument('--train_all_labels', action='store_true',
                        help='Train separate models for all label columns')

    # CatBoost参数
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--l2_leaf_reg', type=float, default=3.0)
    parser.add_argument('--early_stopping_rounds', type=int, default=50)
    parser.add_argument('--auto_class_weights', action='store_true', default=True)
    parser.add_argument('--eval_metric', type=str, default='AUC')

    # 输出
    parser.add_argument('--output_dir', type=str, default='catboost_results')
    parser.add_argument('--verbose', type=int, default=100)
    parser.add_argument('--task_type', type=str, default='CPU')

    return parser.parse_args()


def load_data(train_path, vld_path, label_col):
    """加载数据"""
    print("=" * 70)
    print("📊 Loading Data")
    print("=" * 70)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(vld_path)

    print(f"\nTraining set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")

    # 检查Label列
    label_cols = [col for col in train_df.columns if col.startswith('Label_')]
    print(f"\nFound label columns: {label_cols}")

    if not label_cols:
        raise ValueError("No Label columns found! Expected Label_0, Label_1, etc.")

    if label_col not in train_df.columns:
        raise ValueError(f"Specified label column '{label_col}' not found! Available: {label_cols}")

    print(f"\n✓ Using label column: {label_col}")

    # 打印Label分布
    print(f"\nLabel Distribution for {label_col}:")
    print("\nTraining set:")
    for label in sorted(train_df[label_col].unique()):
        count = (train_df[label_col] == label).sum()
        pct = count / len(train_df) * 100
        print(f"  Class {label}: {count:6d} ({pct:5.1f}%)")

    print("\nTest set:")
    for label in sorted(test_df[label_col].unique()):
        count = (test_df[label_col] == label).sum()
        pct = count / len(test_df) * 100
        print(f"  Class {label}: {count:6d} ({pct:5.1f}%)")

    return train_df, test_df, label_cols


def prepare_data(df, label_col):
    """准备特征和标签"""
    # 所有Label列
    label_cols = [col for col in df.columns if col.startswith('Label_')]

    # 特征 = 所有列 - Label列
    feature_cols = [col for col in df.columns if col not in label_cols]

    X = df[feature_cols]
    y = df[label_col]

    return X, y


def create_params(args):
    """创建CatBoost参数"""
    params = {
        'iterations': args.iterations,
        'learning_rate': args.learning_rate,
        'depth': args.depth,
        'l2_leaf_reg': args.l2_leaf_reg,
        'loss_function': 'Logloss',
        'eval_metric': args.eval_metric,
        'random_seed': 42,
        'verbose': args.verbose,
        'task_type': args.task_type,
    }

    if args.auto_class_weights:
        params['auto_class_weights'] = 'Balanced'

    return params


def train_model(X_train, y_train, X_test, y_test, params, args, label_col):
    """训练模型"""
    print("\n" + "=" * 70)
    print(f"🚀 Training Model for {label_col}")
    print("=" * 70)

    print(f"\nFeatures: {X_train.shape[1]}")

    # 创建Pool
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    # 创建模型
    print("\n🏗️  Creating CatBoostClassifier...")
    model = CatBoostClassifier(**params)

    # 训练
    print("\n🎯 Training...")
    print("-" * 70)

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=True,
        use_best_model=True
    )

    print("\n✅ Training completed!")

    # 最佳迭代
    best_iteration = model.get_best_iteration()
    best_score = model.get_best_score()

    print(f"\n📊 Best Results:")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Best score: {best_score}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型"""
    print("\n" + "=" * 70)
    print("📈 Model Evaluation")
    print("=" * 70)

    results = {}

    # 训练集
    print("\n--- Training Set ---")
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
    results['train'] = train_metrics
    print_metrics(train_metrics)

    # 测试集
    print("\n--- Test Set ---")
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
    results['test'] = test_metrics
    print_metrics(test_metrics)

    # 混淆矩阵
    print("\n--- Confusion Matrix (Test) ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    return results


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """计算指标"""
    metrics = {
        'auroc': float(roc_auc_score(y_true, y_pred_proba)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
    }

    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return metrics


def print_metrics(metrics):
    """打印指标"""
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")


def plot_feature_importance(model, X, output_dir, label_col, top_n=20):
    """绘制特征重要性"""
    print(f"\n📊 Feature Importance for {label_col}...")

    importance = model.get_feature_importance()
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 保存CSV
    csv_path = os.path.join(output_dir, f'feature_importance_{label_col}.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 打印Top N
    print(f"\n  Top {top_n} Features:")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"    {row['feature']}: {row['importance']:.2f}")

    # 绘图
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance - {label_col}')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'feature_importance_{label_col}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {plot_path}")

    return importance_df


def save_results(model, results, output_dir, label_col):
    """保存结果"""
    # 保存模型
    model_path = os.path.join(output_dir, f'catboost_model_{label_col}.cbm')
    model.save_model(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # 保存结果
    results_path = os.path.join(output_dir, f'results_{label_col}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved: {results_path}")


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 70)
    print("  🐱 CatBoost Multi-Label Classification")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {args.output_dir}")

    # 加载数据
    train_df, test_df, label_cols = load_data(args.train_path, args.vld_path, args.label_col)

    # 创建参数
    params = create_params(args)

    # 训练所有Label或单个Label
    if args.train_all_labels:
        print("\n" + "=" * 70)
        print(f"🔄 Training models for all {len(label_cols)} labels")
        print("=" * 70)

        all_results = {}

        for label_col in label_cols:
            print(f"\n\n{'=' * 70}")
            print(f"Processing: {label_col}")
            print("=" * 70)

            # 准备数据
            X_train, y_train = prepare_data(train_df, label_col)
            X_test, y_test = prepare_data(test_df, label_col)

            # 训练
            model = train_model(X_train, y_train, X_test, y_test, params, args, label_col)

            # 评估
            results = evaluate_model(model, X_train, y_train, X_test, y_test)
            all_results[label_col] = results

            # 特征重要性
            plot_feature_importance(model, X_train, args.output_dir, label_col, top_n=20)

            # 保存
            save_results(model, results, args.output_dir, label_col)

        # 保存汇总结果
        summary_path = os.path.join(args.output_dir, 'results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Summary saved: {summary_path}")

        # 打印汇总
        print("\n" + "=" * 70)
        print("  📊 Summary of All Labels")
        print("=" * 70)
        for label_col, results in all_results.items():
            print(f"\n{label_col}:")
            print(f"  Test AUROC: {results['test']['auroc']:.4f}")
            print(f"  Test Accuracy: {results['test']['accuracy']:.4f}")

    else:
        # 只训练一个Label
        X_train, y_train = prepare_data(train_df, args.label_col)
        X_test, y_test = prepare_data(test_df, args.label_col)

        # 训练
        model = train_model(X_train, y_train, X_test, y_test, params, args, args.label_col)

        # 评估
        results = evaluate_model(model, X_train, y_train, X_test, y_test)

        # 特征重要性
        plot_feature_importance(model, X_train, args.output_dir, args.label_col, top_n=20)

        # 保存
        save_results(model, results, args.output_dir, args.label_col)

    # 完成
    print("\n" + "=" * 70)
    print("  ✅ ALL TASKS COMPLETED!")
    print("=" * 70)
    print(f"\nResults saved in: {args.output_dir}/")
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()