# 📷 本地识图工具

调用云端视觉模型，识别本地图片并返回文字描述。

## 功能

- 单张 / 多张 / 整目录识别
- 自定义提问
- **批量并行识别**（`-c` 控制并发数，默认 4）
- **多轮对话**（`-i`，识别后对同一张图连续追问）

## 用法

```bash
# 识别一张图
python vision.py 图片.png

# 识别多张图
python vision.py a.png b.png c.jpg

# 识别整个目录
python vision.py ./截图

# 自定义问题
python vision.py 图.png -q "图里有什么文字？"

# 多轮对话（识别后可连续追问）
python vision.py 图.png -i

# 批量识别 + 控制并行数（默认并行 4，上限 8）
python vision.py ./截图 -c 6
```

## 配置

API Key 在 `.env` 文件里（已 gitignore，不会泄露）：

```
VISION_BASE_URL=https://api.siliconflow.cn/v1/chat/completions
VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
VISION_API_KEY=sk-xxxx
```

当前用的是**硅基流动 SiliconFlow** + **Qwen3-VL-8B**。想换更大的模型（质量更好、稍慢）：

```
VISION_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct
```

换别的厂商（OpenAI、智谱等），改 `.env` 里的三个配置即可。

## 依赖

- Python 3.8+
- `requests`

```bash
pip install requests
```

## 在对话里配合使用

把图片路径告诉 Claude Code，它会调用这个脚本识图后再帮你分析。
