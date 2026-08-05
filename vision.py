#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vision.py — 本地识图工具

调用云端视觉模型识别本地图片，返回文字描述。
默认接入硅基流动 SiliconFlow（Qwen3-VL 系列），也可以通过 .env 里的配置
切换到任意 OpenAI 兼容接口（OpenAI、智谱、Gemini 等）。

用法:
    python vision.py <图片路径> [更多图片...] [-q "你想问的问题"]
    python vision.py <图片目录>

API Key 放在同目录的 .env 文件里（三个配置项）:
    VISION_BASE_URL=https://api.siliconflow.cn/v1/chat/completions
    VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
    VISION_API_KEY=sk-xxxx
"""

import os
import sys
import base64
import argparse
import mimetypes

import requests

# ---------------- 默认配置（硅基流动 Qwen3-VL） ----------------
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QUESTION = "请详细描述这张图片的内容，包括场景、主体、文字和颜色。"


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


def recognize(base_url, model, api_key, image_path, question):
    """发送单张图片到视觉模型，返回描述文字。"""
    content = [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(base_url, json=payload, headers=headers, timeout=120)
    except requests.RequestException as e:
        return f"[网络错误] {e}"

    if resp.status_code != 200:
        return f"[请求失败 HTTP {resp.status_code}] {resp.text[:300]}"

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"[返回格式异常] {str(data)[:300]}"


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


def main():
    parser = argparse.ArgumentParser(description="本地识图工具（云端视觉模型）")
    parser.add_argument("paths", nargs="+", help="图片路径，可多个，也可以是目录")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION, help="要问图片的问题")
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
    print(f"识别 {len(images)} 张图片…")
    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img}")
        print("-" * 44)
        print(recognize(base_url, model, api_key, img, args.question))


if __name__ == "__main__":
    main()
