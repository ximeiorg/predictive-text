#!/bin/bash
# 分步处理维基百科数据

set -e

echo "=========================================="
echo "维基百科数据处理 - 分步执行"
echo "=========================================="

# 步骤1: 解压bz2文件
echo ""
echo "步骤1: 解压 bz2 文件..."
echo "----------------------------------------"

INPUT_BZ2="data/zhwiki-latest-pages-articles.xml.bz2"
OUTPUT_XML="data/zhwiki-latest-pages-articles.xml"

if [ -f "$OUTPUT_XML" ]; then
    SIZE=$(du -h "$OUTPUT_XML" | cut -f1)
    echo "✓ 已存在解压文件: $SIZE"
    read -p "  是否重新解压? (y/N): " choice
    if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
        rm "$OUTPUT_XML"
    else
        echo "  跳过解压步骤"
        SKIP_UNZIP=1
    fi
fi

if [ -z "$SKIP_UNZIP" ]; then
    echo "正在解压 $INPUT_BZ2..."
    echo "预计需要 10-20 分钟..."
    
    # 使用 pv 显示进度
    if command -v pv &> /dev/null; then
        pv "$INPUT_BZ2" | bzip2 -d > "$OUTPUT_XML"
    else
        # 没有 pv，使用标准解压
        echo "提示: 安装 pv 可以显示进度 (brew install pv)"
        bzip2 -dk "$INPUT_BZ2"
        mv "${INPUT_BZ2%.bz2}" "$OUTPUT_XML"
    fi
    
    SIZE=$(du -h "$OUTPUT_XML" | cut -f1)
    echo "✓ 解压完成: $SIZE"
fi

# 步骤2: 处理 XML 提取文本
echo ""
echo "步骤2: 处理 XML 提取文本..."
echo "----------------------------------------"

OUTPUT_TXT="data/wiki_processed.txt"

echo "正在处理 XML 文件..."
echo "输出: $OUTPUT_TXT"

# 使用 Python 处理
python3 scripts/process_wikipedia.py \
    --input "$OUTPUT_XML" \
    --output "$OUTPUT_TXT" \
    "$@"

# 步骤3: 显示统计
echo ""
echo "=========================================="
echo "处理完成!"
echo "=========================================="

if [ -f "$OUTPUT_TXT" ]; then
    SIZE=$(du -h "$OUTPUT_TXT" | cut -f1)
    LINES=$(wc -l < "$OUTPUT_TXT" | tr -d ' ')
    
    echo "输出文件: $OUTPUT_TXT"
    echo "文件大小: $SIZE"
    echo "文章数量: $LINES"
    
    echo ""
    echo "前5行预览:"
    head -5 "$OUTPUT_TXT"
fi