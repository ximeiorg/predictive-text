# 维基百科数据处理方案

## 方案1: 分步处理（推荐）

先解压，再处理，进度清晰：

```bash
# 1. 解压 bz2 文件（10-20分钟）
bzip2 -dk data/zhwiki-latest-pages-articles.xml.bz2
# 或使用 pv 显示进度
pv data/zhwiki-latest-pages-articles.xml.bz2 | bzip2 -d > data/zhwiki-latest-pages-articles.xml

# 2. 处理 XML 提取文本（进度条显示）
uv run python scripts/process_wikipedia.py \
  --input data/zhwiki-latest-pages-articles.xml \
  --output data/wiki_processed.txt \
  --max-articles 100000
```

## 方案2: 使用脚本自动化

```bash
bash scripts/process_wiki_step_by_step.sh --max-articles 100000
```

这个脚本会：
1. 检查是否已解压
2. 自动解压（如果需要）
3. 处理 XML 提取文本
4. 显示统计信息

## 方案3: 直接处理压缩文件

```bash
uv run python scripts/process_wikipedia.py \
  --input data/zhwiki-latest-pages-articles.xml.bz2 \
  --output data/wiki_processed.txt
```

**注意**: 直接处理压缩文件进度显示可能不准确

## 文件大小预估

- 压缩文件: 3.0 GB
- 解压后 XML: ~10 GB
- 提取的文本: ~500-800 MB（取决于过滤条件）

## 进度说明

解压后的处理会显示：

```
Parsing XML:  12345pages [01:23, 148.52pages/s, valid=8234]
Writing articles:  8234articles [01:25, 96.84articles/s, chars=15.2M, size=45.3MB]
```

- **Parsing XML**: 解析进度和有效文章数
- **Writing articles**: 写入进度和文件大小

## 参数说明

```bash
--input          输入文件（.xml 或 .xml.bz2）
--output         输出文本文件
--max-articles   最大文章数（测试用）
--min-length     最小文本长度（默认50）
--max-length     最大文本长度（默认10000）
```

## 建议

1. **首次处理**: 使用 `--max-articles 10000` 测试
2. **完整处理**: 去掉 `--max-articles` 或设置更大的值
3. **磁盘空间**: 确保有 15GB 以上可用空间