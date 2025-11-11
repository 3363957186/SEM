#!/bin/zsh

SOURCE="/Users/e/Downloads/SCAN_MRI"
OUTPUT="data/nacc_subset"
SAMPLE_SIZE=2000

echo "=== NACC子集处理 ==="
echo "将处理 $SAMPLE_SIZE 个样本"
echo ""

cd "$SOURCE"

echo "📁 选择样本..."
SAMPLES=($(find . -maxdepth 1 -name "*.zip" -type f | head -$SAMPLE_SIZE))

echo "开始处理 ${#SAMPLES[@]} 个文件"
echo ""
count=0
for zip_file in $SAMPLES; do
    count=$((count + 1))
    echo "[$count/${#SAMPLES[@]}] 处理: $(basename $zip_file)"
    
    # 解压到临时目录
    temp_dir="$OUTPUT/temp_$(basename "$zip_file" .zip)"
    mkdir -p "$temp_dir"           # 创建多级父目录
    unzip -oq "$zip_file" -d "$temp_dir"

    
    # 这里添加DICOM→NIfTI→Embedding的处理
    # ...
    
    # 清理临时文件
    # rm -rf "$temp_dir"
    
    # 显示进度
    if (( count % 10 == 0 )); then
        echo "  已完成 $count/$SAMPLE_SIZE"
    fi
done

echo ""
echo "=== 完成 ==="
du -sh "$OUTPUT"
