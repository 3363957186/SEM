#!/bin/bash

# ========== 配置参数 ==========
TRAIN_PATH="data/train_data_X.csv"
VLD_PATH="data/test_data_X.csv"
OUTPUT_DIR="catboost_results"

# 选择训练模式
# 模式1: 训练单个Label（快速）
# 模式2: 训练所有Label（完整）
TRAIN_ALL_LABELS=true  # true=训练所有, false=只训练一个

# 如果只训练一个Label，选择哪个
LABEL_COL="Label_"  # Label_0, Label_1, Label_2, Label_3

# CatBoost参数
ITERATIONS=128
LEARNING_RATE=0.001
DEPTH=3
EARLY_STOPPING=50

echo "Configuration:"
echo "  Train data: $TRAIN_PATH"
echo "  Validation data: $VLD_PATH"
echo "  Train all labels: $TRAIN_ALL_LABELS"
if [ "$TRAIN_ALL_LABELS" = false ]; then
    echo "  Target label: $LABEL_COL"
fi
echo "  Iterations: $ITERATIONS"
echo ""

# 构建命令
CMD="python dev/train_catboost.py \
    --train_path $TRAIN_PATH \
    --vld_path $VLD_PATH \
    --iterations $ITERATIONS \
    --learning_rate $LEARNING_RATE \
    --depth $DEPTH \
    --early_stopping_rounds $EARLY_STOPPING \
    --auto_class_weights \
    --eval_metric AUC \
    --output_dir $OUTPUT_DIR \
    --task_type CPU \
    --verbose 100"

# 添加label参数
if [ "$TRAIN_ALL_LABELS" = true ]; then
    CMD="$CMD --train_all_labels"
else
    CMD="$CMD --label_col $LABEL_COL"
fi

# 执行
eval $CMD

# ========== 检查结果 ==========
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "  ✅ SUCCESS!"
    echo "======================================================================"
    echo ""
    echo "Results saved in: $OUTPUT_DIR/"
    echo ""
    if [ "$TRAIN_ALL_LABELS" = true ]; then
        echo "Generated files for each label:"
        echo "  - catboost_model_Label_X.cbm"
        echo "  - results_Label_X.json"
        echo "  - feature_importance_Label_X.csv/png"
        echo "  - results_summary.json (all labels)"
    else
        echo "Generated files:"
        echo "  - catboost_model_$LABEL_COL.cbm"
        echo "  - results_$LABEL_COL.json"
        echo "  - feature_importance_$LABEL_COL.csv/png"
    fi
    echo ""
    echo "View results:"
    if [ "$TRAIN_ALL_LABELS" = true ]; then
        echo "  cat $OUTPUT_DIR/results_summary.json"
    else
        echo "  cat $OUTPUT_DIR/results_$LABEL_COL.json"
    fi
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "  ❌ FAILED!"
    echo "======================================================================"
    echo ""
    exit 1
fi