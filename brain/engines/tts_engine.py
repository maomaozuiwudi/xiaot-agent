"""火山引擎 TTS 引擎 — 声音复刻"""
import base64, json, os, uuid
import urllib.request, urllib.error, ssl
from pathlib import Path


def _load_volc_key():
    """从文件读取火山引擎 API Key"""
    key_path = Path.home() / "voice_clone" / ".volc_voice_key"
    try:
        return key_path.read_text().strip()
    except Exception:
        return os.environ.get("VOLC_VOICE_KEY", "")


def synthesize(text, voice_id="S_wHXLNCs52", speed_ratio=1.1, output_path=None):
    """
    调用火山引擎声音复刻 TTS 将文本转为语音

    Args:
        text: 要转语音的文本
        voice_id: 声音复刻 Voice ID
        speed_ratio: 语速 (1.0=正常, 1.1=稍快)
        output_path: 输出音频路径，None 时自动生成
    Returns:
        str: 输出文件路径
    """
    api_key = _load_volc_key()
    if not api_key:
        raise ValueError("火山引擎 API Key 未配置")

    url = "https://openspeech.bytedance.com/api/v1/tts"

    request_json = {
        "app": {"cluster": "volcano_icl"},
        "voice": {
            "voice_type": "voice_clone",
            "voice_id": voice_id,
            "speed_ratio": speed_ratio,
        },
        "audio": {
            "encoding": "mp3",
            "sample_rate": 24000,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": "plain",
            "operation": "query",
        },
    }

    body = json.dumps(request_json).encode("utf-8")
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", api_key)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    resp = urllib.request.urlopen(req, context=ctx, timeout=60)
    result = json.loads(resp.read())

    if result.get("code") != 3000:
        raise RuntimeError(f"TTS 调用失败: code={result.get('code')}, message={result.get('message', '')}")

    audio_data = base64.b64decode(result["data"])

    if output_path is None:
        output_path = f"output/tts_{uuid.uuid4().hex[:8]}.mp3"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_data)

    return output_path
