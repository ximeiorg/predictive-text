# 手机端部署指南

## 方案对比

| 方案 | 性能 | 内存 | int8 | int4 | 推荐度 |
|-----|------|-----|------|------|-------|
| **MNN** | 最快 | 最小 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| ONNX Runtime | 中等 | 中等 | ✅ | ❌ | ⭐⭐⭐ |
| PyTorch Mobile | 慢 | 大 | ✅ | ❌ | ⭐⭐ |

**推荐 MNN**：阿里开源，专为移动端优化，输入法广泛使用。

---

## 导出流程

### 1. 安装依赖

```bash
# 基础依赖
uv sync

# 移动端导出依赖
uv sync --extra mobile

# 或手动安装
pip install onnx onnxruntime
```

### 2. 安装 MNN 工具

```bash
# 方式一：预编译版本 (推荐)
# 从 https://github.com/alibaba/MNN/releases 下载

# 方式二：源码编译
git clone https://github.com/alibaba/MNN.git
cd MNN
cmake -DMNN_BUILD_CONVERTER=ON .
make -j4
cp tools/converter/MNNConvert /usr/local/bin/
```

### 3. 导出模型

```bash
# 导出 MNN int8 模型 (推荐)
uv run src/export_mobile.py \
    --checkpoint output/best_model.pt \
    --vocab data/vocab.json \
    --output-dir mobile \
    --format mnn \
    --quant-bits 8

# 导出 int4 模型 (更小体积)
uv run src/export_mobile.py --quant-bits 4

# 导出所有格式
uv run src/export_mobile.py --format all

# 仅导出 ONNX (MNN工具不可用时)
uv run src/export_mobile.py --format onnx
```

### 4. 校准量化 (效果更好)

```bash
# 使用校准数据的静态量化
uv run scripts/quantize_with_calibration.py \
    --checkpoint output/best_model.pt \
    --train-data data/train.bin \
    --vocab data/vocab.json \
    --method onnx_static \
    --calibration-samples 200
```

---

## 导出文件说明

导出后在 `mobile/` 目录生成：

```
mobile/
├── model_q8.mnn      # MNN int8 模型 (推荐)
├── model_q4.mnn      # MNN int4 模型 (可选)
├── model.onnx        # ONNX float32 模型
├── model_q8.onnx     # ONNX int8 模型
├── vocab.json        # 词表 JSON
├── vocab.txt         # 词表文本
├── manifest.json     # 元信息
└── README.md         # 部署说明
```

---

## Android 部署 (MNN)

### 1. 添加 MNN 依赖

```gradle
// app/build.gradle
dependencies {
    implementation 'com.alibaba.android:mnn:1.1.0'
}
```

### 2. 加载模型

```kotlin
import com.alibaba.android.mnn.MNNInterpreter

class InputMethodModel(context: Context) {
    private val interpreter: MNNInterpreter
    
    init {
        // 从 assets 加载
        val modelPath = context.assets.open("model_q8.mnn").let {
            val file = File(context.cacheDir, "model_q8.mnn")
            it.copyTo(file.outputStream())
            file.absolutePath
        }
        interpreter = MNNInterpreter.createFromFile(modelPath)
    }
    
    fun predict(inputIds: IntArray): IntArray {
        val session = interpreter.createSession()
        val inputTensor = interpreter.createInputTensor(inputIds)
        
        interpreter.runSession(session, inputTensor)
        
        val outputTensor = interpreter.getOutputTensor(session, 0)
        val logits = outputTensor.getFloatData()
        
        // 取最后一个位置的 top-5
        val lastLogits = logits.takeLast(vocabSize)
        return lastLogits.indices.sortedByDescending { lastLogits[it] }.take(5)
    }
}
```

### 3. 分词集成

```kotlin
class Vocabulary(context: Context) {
    private val word2id: Map<String, Int>
    private val id2word: Map<Int, String>
    
    init {
        val json = context.assets.open("vocab.json").bufferedReader().use { it.readText() }
        word2id = JSONObject(json).toMap()
        id2word = word2id.entries.associate { (k, v) -> v to k }
    }
    
    fun encode(text: String): IntArray {
        // 使用 jieba 或简化分词
        val words = text.split("")  // 或集成 jieba-android
        return words.map { word2id[it] ?: 3 }.toIntArray()
    }
    
    fun decode(ids: IntArray): String {
        return ids.map { id2word[it] ?: "[UNK]" }.joinToString("")
    }
}
```

---

## iOS 部署 (MNN)

### 1. 添加 MNN 框架

```swift
// Podfile
pod 'MNN', '~> 1.1.0'
```

### 2. 加载模型

```swift
import MNN

class InputMethodModel {
    private let interpreter: MNNInterpreter
    private let vocab: [String: Int]
    
    init() {
        guard let modelPath = Bundle.main.path(forResource: "model_q8", ofType: "mnn") else {
            fatalError("Model not found")
        }
        
        interpreter = MNNInterpreter(file: modelPath)
        
        // 加载词表
        guard let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "json") else {
            fatalError("Vocab not found")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] ?? [:()
    }
    
    func predict(inputText: String) -> [String] {
        let inputIds = encode(inputText)
        
        let session = interpreter.createSession()
        let inputTensor = interpreter.createInputTensor(inputIds)
        
        interpreter.runSession(session, inputTensor)
        
        let output = interpreter.getOutputTensor(session, 0)
        let logits = output.getData() as! [Float]
        
        // 获取 top-5 建议词
        let lastPosLogits = logits.suffix(vocabSize)
        let topIndices = lastPosLogits.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(5)
            .map { $0.offset }
        
        return topIndices.compactMap { id2word[$0] }
    }
}
```

---

## 性能对比

以 6M 参数模型 (vocab=8192, hidden=256, layers=4) 为例：

| 格式 | 模型大小 | 推理时间 | 内存占用 |
|-----|---------|---------|---------|
| float32 | ~24 MB | ~50ms | ~100 MB |
| int8 (MNN) | ~6 MB | ~20ms | ~30 MB |
| int4 (MNN) | ~3 MB | ~25ms | ~20 MB |

**int8量化效果**：体积减少75%，速度提升2-3倍，精度损失<2%

---

## 输入法集成建议

### 架构设计

```
┌─────────────────────────────────────────────┐
│                 输入法应用                    │
├─────────────────────────────────────────────┤
│  用户输入: "今天天气"                         │
│      ↓                                      │
│  ┌─────────────────┐                        │
│  │   分词模块       │                        │
│  │  jieba / 简化   │                        │
│  └─────────────────┘                        │
│      ↓ [1, 234, 567, 890]                   │
│  ┌─────────────────┐                        │
│  │   模型推理       │                        │
│  │  MNN / ONNX     │                        │
│  └─────────────────┘                        │
│      ↓ logits [vocab_size]                  │
│  ┌─────────────────┐                        │
│  │   后处理         │                        │
│  │  top-k + 解码   │                        │
│  └─────────────────┘                        │
│      ↓                                      │
│  建议词: ["很好", "不错", "晴朗", ...]       │
└─────────────────────────────────────────────┘
```

### 优化建议

1. **缓存模型**: 应用启动时加载，避免每次输入重新加载
2. **批量预测**: 用户输入多个字时，缓存中间结果
3. **延迟加载**: 首次使用时才加载模型
4. **热更新**: 支持从服务器下载新模型

---

## 常见问题

**Q: MNNConvert 找不到？**
A: 从 MNN releases 下载或源码编译

**Q: int4 量化精度损失？**
A: 对于词语联想任务，int4 通常可接受。建议测试对比。

**Q: Android JNI 错误？**
A: 确保 MNN AAR 版本与模型版本匹配

**Q: iOS 编译错误？**
A: 检查 MNN framework 是否正确链接