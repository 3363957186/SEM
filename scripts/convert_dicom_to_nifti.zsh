#!/bin/zsh

INPUT_DIR="data/nacc_subset"
OUTPUT_DIR="data/nacc_nifti"
LOG_FILE="data/conversion_log.txt"

echo "=== DICOM → NIfTI 转换 (修复版) ===" | tee "$LOG_FILE"
mkdir -p "$OUTPUT_DIR"

# 获取所有患者目录
PATIENTS=($(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort))
TOTAL=${#PATIENTS[@]}

echo "找到 $TOTAL 个患者" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

count=0
success=0
failed=0

# 记录已处理的患者ID（避免重复）
declare -A processed

for patient_dir in $PATIENTS; do
    count=$((count + 1))
    patient_id=$(basename "$patient_dir")
    
    echo "[$count/$TOTAL] 处理: $patient_id" | tee -a "$LOG_FILE"
    
    # 简单的患者ID（用于检查是否已处理）
    # 提取主要ID部分（例如从长文件名提取NACC号）
    simple_id=$(echo "$patient_id" | grep -o "NACC[0-9]*" | head -1)
    if [ -z "$simple_id" ]; then
        simple_id="$patient_id"
    fi
    
    # 检查是否已处理（宽松匹配）
    if [ ${processed[$simple_id]+_} ]; then
        echo "  ⏭️  已处理过此患者，跳过" | tee -a "$LOG_FILE"
        continue
    fi
    
    # 检查DICOM文件数量
    dcm_count=$(find "$patient_dir" -name "*.dcm" 2>/dev/null | wc -l | tr -d ' ')
    echo "  📊 DICOM文件数: $dcm_count" | tee -a "$LOG_FILE"
    
    if [ $dcm_count -eq 0 ]; then
        echo "  ⚠️  没有DICOM文件，跳过" | tee -a "$LOG_FILE"
        failed=$((failed + 1))
        continue
    fi
    
    # 转换
    echo "  🔄 转换中..." | tee -a "$LOG_FILE"
    
    # 记录转换前的文件数
    before_count=$(find "$OUTPUT_DIR" -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l | tr -d ' ')
    
    # 执行转换
    temp_output=$(mktemp)
    dcm2niix -o "$OUTPUT_DIR" -f "$patient_id" -z y -m y "$patient_dir" > "$temp_output" 2>&1
    convert_status=$?
    
    # 记录转换后的文件数
    after_count=$(find "$OUTPUT_DIR" -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l | tr -d ' ')
    new_files=$((after_count - before_count))
    
    if [ $convert_status -eq 0 ] && [ $new_files -gt 0 ]; then
        success=$((success + 1))
        processed[$simple_id]=1
        echo "  ✅ 成功 (生成 $new_files 个文件)" | tee -a "$LOG_FILE"
    else
        failed=$((failed + 1))
        echo "  ❌ 失败或未生成文件" | tee -a "$LOG_FILE"
        echo "  dcm2niix输出:" | tee -a "$LOG_FILE"
        head -5 "$temp_output" | tee -a "$LOG_FILE"
    fi
    rm "$temp_output"
    
    # 每10个显示进度和当前状态
    if (( count % 10 == 0 )); then
        echo "  📈 进度: 成功=$success, 失败=$failed, 总文件数=$after_count" | tee -a "$LOG_FILE"
        echo "  💾 当前大小: $(du -sh "$OUTPUT_DIR" | cut -f1)" | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

echo "=== 完成 ===" | tee -a "$LOG_FILE"
echo "总患者数: $TOTAL" | tee -a "$LOG_FILE"
echo "成功转换: $success" | tee -a "$LOG_FILE"
echo "失败: $failed" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "输出目录大小:" | tee -a "$LOG_FILE"
du -sh "$OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "生成的NIfTI文件数:" | tee -a "$LOG_FILE"
nifti_count=$(find "$OUTPUT_DIR" -name "*.nii.gz" -o -name "*.nii" 2>/dev/null | wc -l)
echo "$nifti_count" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "示例文件:" | tee -a "$LOG_FILE"
find "$OUTPUT_DIR" -name "*.nii.gz" 2>/dev/null | head -5 | tee -a "$LOG_FILE"

EOFSCRIPT

