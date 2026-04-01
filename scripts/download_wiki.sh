#!/bin/bash
# 维基百科数据下载脚本 - 支持断点续传

URL="https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
OUTPUT="zhwiki-latest-pages-articles.xml.bz2"

echo "=== 维基百科中文数据下载 ==="
echo "文件: $OUTPUT"
echo "支持断点续传"
echo ""

# 方案1: wget 断点续传
download_with_wget() {
    echo "使用 wget 下载..."
    wget -c \
         --retry-connrefused \
         --waitretry=1 \
         --read-timeout=20 \
         --timeout=15 \
         --tries=0 \
         --no-check-certificate \
         -O "$OUTPUT" \
         "$URL"
}

# 方案2: curl 断点续传
download_with_curl() {
    echo "使用 curl 下载..."
    curl -L -C - \
         --retry 999 \
         --retry-max-time 0 \
         --connect-timeout 30 \
         --max-time 600 \
         --insecure \
         -o "$OUTPUT" \
         "$URL"
}

# 方案3: aria2c 多线程下载（推荐）
download_with_aria2c() {
    if command -v aria2c &> /dev/null; then
        echo "使用 aria2c 多线程下载..."
        aria2c -c \
               -x 16 \
               -s 16 \
               --check-certificate=false \
               -d . \
               -o "$OUTPUT" \
               "$URL"
    else
        echo "aria2c 未安装，使用 wget..."
        download_with_wget
    fi
}

# 检查文件是否存在
if [ -f "$OUTPUT" ]; then
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "发现已下载文件: $SIZE"
    echo "将继续下载..."
fi

# 选择下载方式
echo ""
echo "选择下载方式:"
echo "1. wget (推荐)"
echo "2. curl"
echo "3. aria2c (多线程，需要安装)"
echo ""
read -p "请选择 [1-3, 默认1]: " choice

case $choice in
    2) download_with_curl ;;
    3) download_with_aria2c ;;
    *) download_with_wget ;;
esac

# 检查下载结果
if [ -f "$OUTPUT" ]; then
    FINAL_SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo ""
    echo "下载完成！"
    echo "文件大小: $FINAL_SIZE"
    echo "文件位置: $(pwd)/$OUTPUT"

    # 验证文件
    echo ""
    echo "验证文件完整性..."
    bzip2 -t "$OUTPUT" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ 文件完整性验证通过"
    else
        echo "✗ 文件可能损坏，请重新下载"
    fi
else
    echo "下载失败"
fi