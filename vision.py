#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision.py — 本地识图工具

调用云端视觉模型识别本地图片，返回文字描述。
默认接入硅基流动 SiliconFlow（Qwen3-VL 系列），也可以通过 .env 里的配置
切换到任意 OpenAI 兼容接口（OpenAI、智谱、Gemini 等）。

用法:
    python vision.py <图片路径> [更多图片...] [-q "你想问的问题"]
    python vision.py <图片目录>                      # 批量识别（默认并行 4）
    python vision.py <图片目录> -c 6                 # 手动指定并行数
    python vision.py <图片> -i                       # 识别后进入多轮对话（可追问）
    python vision.py <图片> --json                   # 返回结构化 JSON（程序可直接解析）
    python vision.py <图片> -o 结果.json             # 把结果保存到文件

API Key 放在同目录的 .env 文件里（三个配置项）:
    VISION_BASE_URL=https://api.siliconflow.cn/v1/chat/completions
    VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
    VISION_API_KEY=sk-xxxx
"""

import os
import sys
import json
import time
import base64
import argparse
import mimetypes
from concurrent.futures import ThreadPoolExecutor

import requests

# Windows 控制台默认 GBK 编码，遇到 emoji 等字符会直接崩溃，这里强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------- 默认配置（硅基流动 Qwen3-VL） ----------------
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QUESTION = "请详细描述这张图片的内容，包括场景、主体、文字和颜色。"
DEFAULT_JSON_QUESTION = (
    "请用 JSON 格式描述这张图片。只输出一个 JSON 对象，不要输出任何其他内容，字段如下：\n"
    'scene(场景描述, string), subjects(主要主体, string数组), '
    'text(图片中出现的所有文字, string数组), colors(主要颜色, string数组), '
    'summary(一句话总结, string)'
)
MAX_CONCURRENCY = 8  # 并行数上限，防止把 API 打爆


def load_config():
    """从 .env 读取配置，返回 (base_url, model, api_key)。"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip().strip('"').strip("'")

    base_url = env.get("VISION_BASE_URL", DEFAULT_BASE_URL)
    model = env.get("VISION_MODEL", DEFAULT_MODEL)
    api_key = env.get("VISION_API_KEY") or env.get("ZHIPU_API_KEY") or os.environ.get("ZHIPU_API_KEY", "")
    return base_url, model, api_key


def to_data_url(path: str) -> str:
    """把本地图片转成 base64 的 data URL。"""
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def chat_once(base_url, model, api_key, messages, retries=2, json_mode=False):
    """发送一整套消息（支持多轮历史）到视觉模型，返回 (回答文本, usage)。

    网络错误会带退避重试（最多 retries 次）。json_mode=True 时请求带
    response_format 强制模型输出 JSON。
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(base_url, json=payload, headers=headers, timeout=120)
            break
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))  # 2s / 4s 退避
    else:
        return f"[网络错误] {last_err}", None

    if resp.status_code != 200:
        return f"[请求失败 HTTP {resp.status_code}] {resp.text[:300]}", None

    try:
        resp.encoding = "utf-8"  # 强制 UTF-8，避免 chardet 误判成 GBK 导致中文乱码
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")
        return content, usage
    except Exception as e:
        return f"[返回格式异常] {e}", None


def try_parse_json(text):
    """尝试把模型回答解析成 JSON；容忍 ```json 代码块和前后杂散文本。"""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def recognize(base_url, model, api_key, image_path, question, json_mode=False):
    """发送单张图片到视觉模型，返回描述文字（单轮）。

    json_mode=True 时要求返回 JSON；若 response_format 不被支持或模型没听话，
    自动降级重试一次，仍失败则返回 {"error": ..., "raw": 原始回答}。
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
        ],
    }]
    if json_mode:
        answer, _ = chat_once(base_url, model, api_key, messages, json_mode=True)
        parsed = try_parse_json(answer)
        if parsed is not None:
            return parsed
        # 降级：不带 response_format 再试一次，仅靠 prompt 约束
        answer, _ = chat_once(base_url, model, api_key, messages, json_mode=False)
        parsed = try_parse_json(answer)
        if parsed is not None:
            return parsed
        return {"error": "模型未返回合法 JSON", "raw": answer}
    answer, _ = chat_once(base_url, model, api_key, messages)
    return answer


def read_line(prompt=""):
    """读一行输入，兼容两种 stdin 编码：UTF-8（管道/脚本喂入）和 GBK（Windows 控制台打字）。"""
    if prompt:
        print(prompt, end="", flush=True)
    raw = sys.stdin.buffer.readline()
    if not raw:  # EOF
        return None
    raw = raw.rstrip(b"\r\n")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:  # GBK 控制台输入会解码失败，退回系统编码
        return raw.decode(sys.stdin.encoding or "gbk", errors="replace")


def interactive_chat(base_url, model, api_key, image_path):
    """对单张图片进行多轮对话：图片只在第一轮发送，之后带上历史继续追问。"""
    print(f"进入对话模式：{image_path}")
    print("直接回车 = 默认描述；输入 q / exit 退出；Ctrl+C 也可退出。")
    history = []  # 已完成轮次的 user/assistant 消息
    while True:
        try:
            q = read_line("💬 问题: ")
        except (EOFError, KeyboardInterrupt):
            break
        if q is None:  # 输入流结束
            break
        q = q.strip()
        if q.lower() in ("q", "exit", "quit"):
            break
        q = q or DEFAULT_QUESTION

        content = [{"type": "text", "text": q}]
        if not history:  # 第一轮才带上图片，之后靠历史上下文
            content.append({"type": "image_url", "image_url": {"url": to_data_url(image_path)}})

        messages = history + [{"role": "user", "content": content}]
        print("-" * 44)
        answer, _ = chat_once(base_url, model, api_key, messages)
        print(answer)

        history.append({"role": "user", "content": content})
        history.append({"role": "assistant", "content": answer})


def collect_images(paths):
    """收集所有图片路径，支持文件、目录、通配。"""
    images = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                mime = mimetypes.guess_type(name)[0]
                if mime and mime.startswith("image/"):
                    images.append(os.path.join(p, name))
        elif os.path.isfile(p):
            images.append(p)
        else:
            print(f"警告: 路径不存在，跳过: {p}")
    return images


def save_results(output, results, json_mode):
    """把 [(图片路径, 结果), ...] 保存到文件。

    .json 结尾 → JSON 数组；其他（.md/.txt）→ 人类可读文本。
    """
    if output.endswith(".json"):
        data = [{"image": img, "result": res} for img, res in results]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(output, "w", encoding="utf-8") as f:
            for img, res in results:
                f.write(f"## {img}\n\n")
                if json_mode:
                    f.write(json.dumps(res, ensure_ascii=False, indent=2))
                else:
                    f.write(str(res))
                f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="本地识图工具（云端视觉模型）")
    parser.add_argument("paths", nargs="+", help="图片路径，可多个，也可以是目录")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION, help="要问图片的问题")
    parser.add_argument("-c", "--concurrency", type=int, default=4,
                        help=f"并行识别数（默认 4，上限 {MAX_CONCURRENCY}）")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="识别后进入多轮对话模式（只对第一张图生效）")
    parser.add_argument("--json", action="store_true",
                        help="要求模型返回结构化 JSON（程序可直接解析）")
    parser.add_argument("-o", "--output",
                        help="把结果保存到文件（.json 存 JSON 数组，.md/.txt 存文本）")
    args = parser.parse_args()

    base_url, model, api_key = load_config()
    if not api_key:
        print("错误: 未找到 API Key。请在 vision/.env 里设置 ZHIPU_API_KEY 或 VISION_API_KEY。")
        sys.exit(1)

    images = collect_images(args.paths)
    if not images:
        print("错误: 没有找到任何图片。")
        sys.exit(1)

    print(f"模型: {model}")

    if args.interactive:
        if args.json:
            print("注意: 对话模式不支持 --json，已忽略。")
        if len(images) > 1:
            print("注意: 对话模式只对第一张图片生效，其余忽略。")
        interactive_chat(base_url, model, api_key, images[0])
        return

    # --json 时替换/追加 JSON 输出约束
    if args.json:
        question = DEFAULT_JSON_QUESTION if args.question == DEFAULT_QUESTION \
            else f"{args.question}\n请只用 JSON 对象格式回答，不要输出任何其他文字。"
    else:
        question = args.question

    concurrency = max(1, min(args.concurrency, MAX_CONCURRENCY))
    print(f"识别 {len(images)} 张图片（并行 {concurrency}）…")

    def safe_recognize(path):
        """包一层，单张失败不影响整批。"""
        try:
            return recognize(base_url, model, api_key, path, question, json_mode=args.json)
        except Exception as e:
            return {"error": f"[处理异常] {e}"} if args.json else f"[处理异常] {e}"

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, (img, result) in enumerate(
            zip(images, ex.map(safe_recognize, images)), 1
        ):
            print(f"\n[{i}/{len(images)}] {img}")
            print("-" * 44)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(result)
            results.append((img, result))

    if args.output:
        save_results(args.output, results, args.json)
        print(f"\n已保存结果到: {args.output}")


if __name__ == "__main__":
    main()
