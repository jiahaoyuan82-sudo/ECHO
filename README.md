# ECHO 谣言与情绪传播分析（网页版）

在网页里输入一条帖子和它下面的回复，ECHO 会给出三样结果：

- 这段对话**像不像谣言**：15 个 ECHO 模型的平均概率和投票结果。
- 每条帖子的**主要情绪**：愤怒、厌恶、恐惧、喜悦、悲伤、惊讶或中性。
- 每条回复**最受哪条帖子影响**，并画出回复树。

所有计算都在浏览器里完成，输入的内容不会上传到服务器。

## 用到的模型

| 用途 | 模型 |
|---|---|
| 情绪识别 | SamLowe/roberta-base-go_emotions（ONNX 量化版） |
| 句向量 | sentence-transformers/all-MiniLM-L6-v2（Xenova ONNX 版） |
| 中译英（可选） | Helsinki-NLP/opus-mt-zh-en（Xenova ONNX 版） |
| 谣言判断 | 论文主实验的 15 个 ECHO 模型（权重放在 `model/` 文件夹） |

运行库是 transformers.js 3.7.6，从 jsDelivr 加载。

## 文件说明

- `index.html`、`style.css`、`app.js`：网页界面。
- `echo.js`：ECHO 模型的 JavaScript 实现，计算方式和论文代码相同。
- `model/echo_weights.bin`、`model/echo_weights.json`：15 个 ECHO 模型的权重，以及每个模型的判断阈值。
- `tools/convert.py`：把 `.pt` 权重转换成上面两个文件的脚本。

本项目是研究原型，结果仅供参考，不能代替事实核查。
