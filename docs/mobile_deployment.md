# 手机端部署指南

## 方案对比

| 方案 | 性能 | 内存 | int8 | 推荐度 |
|-----|------|-----|------|-------|
| **ONNX Runtime** | 快 | 小 | ✅ | ⭐⭐⭐⭐⭐ |
| PyTorch Mobile | 中等 | 中等 | ✅ | ⭐⭐⭐ |

**推荐 ONNX Runtime**：跨平台支持好，性能优秀，易于集成。

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

### 2. 导出模型

```bash
# 导出 ONNX 模型
uv run src/export_mobile.py \
    --checkpoint output/best_model.pt \
    --output-dir mobile

# 指定模型尺寸导出
uv run src/export_mobile.py --model-size small
```

### 3. 校准量化 (效果更好)

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
├── model.onnx        # ONNX float32 模型
├── vocab.json        # 词表 JSON
├── vocab.txt         # 词表文本
├── manifest.json     # 元信息
└── README.md         # 部署说明
```

---

## Android 部署 (ONNX Runtime)

### 1. 添加 ONNX Runtime 依赖

```gradle
// app/build.gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
}
```

### 2. 加载模型

```kotlin
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class InputMethodModel(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    
    init {
        // 从 assets 加载
        val modelPath = context.assets.open("model.onnx").let {
            val file = File(context.cacheDir, "model.onnx")
            it.copyTo(file.outputStream())
            file.absolutePath
        }
        session = env.createSession(modelPath)
    }
    
    fun predict(inputIds: IntArray): IntArray {
        val inputTensor = OnnxTensor.createTensor(env, inputIds.toLongArray())
        val results = session.run(mapOf("input_ids" to inputTensor))
        
        val logits = results.get(0).value as Array<FloatArray>
        
        // 取最后一个位置的 top-5
        val lastLogits = logits[0].last()
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

## iOS 部署 (ONNX Runtime)

### 1. 添加 ONNX Runtime 框架

```swift
// Podfile
pod 'onnxruntime-objc', '~> 1.16.0'
```

### 2. 加载模型

```swift
import onnxruntime_objc

class InputMethodModel {
    private let session: ORTSession
    private let env: ORTEnvironment
    private let vocab: [String: Int]
    
    init() {
        env = ORTEnvironment()
        
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx") else {
            fatalError("Model not found")
        }
        
        session = try! ORTSession(path: modelPath, environment: env)
        
        // 加载词表
        guard let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "json") else {
            fatalError("Vocab not found")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] ?? [:()
    }
    
    func predict(inputText: String) -> [String] {
        let inputIds = encode(inputText)
        
        let inputTensor = try! ORTValue(tensorData: inputIds.toData(), 
                                         elementType: .int64, 
                                         shape: [1, inputIds.count])
        
        let outputs = try! session.run(withInputs: ["input_ids": inputTensor], 
                                        outputNames: ["logits"])
        
        let logits = outputs["logits"]!.tensorData as! [Float]
        
        // 获取 top-5 建议词
        let topIndices = logits.enumerated()
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
| float32 | ~24 MB | ~30ms | ~80 MB |
| int8 (ONNX) | ~8 MB | ~15ms | ~40 MB |

**int8量化效果**：体积减少66%，速度提升2倍，精度损失<2%

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
│  │  ONNX Runtime   │                        │
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

**Q: ONNX Runtime 性能如何？**
A: 在移动端性能优秀，支持 GPU 加速

**Q: int8 量化精度损失？**
A: 对于词语联想任务，int8 通常可接受。建议测试对比。

**Q: Android JNI 错误？**
A: 确保 ONNX Runtime 版本正确，检查 ABI 配置

**Q: iOS 编译错误？**
A: 检查 CocoaPods 安装，确保 framework 正确链接