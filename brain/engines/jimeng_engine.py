"""即梦AI文生图引擎 — 火山引擎V4签名调用"""
import json, hashlib, hmac, urllib.request, datetime, time
from pathlib import Path

_HOST = "visual.volcengineapi.com"
_REGION = "cn-north-1"
_SERVICE = "cv"

def _load_keys():
    """从文件读取即梦 AK/SK"""
    key_file = Path.home() / ".jimeng_key"
    if key_file.exists():
        lines = key_file.read_text().strip().splitlines()
        if len(lines) >= 2:
            return lines[0].strip(), lines[1].strip()
    # fallback: from mem
    return "", ""

def _save_keys(ak, sk):
    """保存即梦 AK/SK"""
    key_file = Path.home() / ".jimeng_key"
    key_file.write_text(f"{ak}\n{sk}")
    key_file.chmod(0o600)

def _sign(method, url_path, query, body, ak, sk):
    """火山引擎HMAC-SHA256 V4签名"""
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    canonical_headers = f"content-type:application/json\nhost:{_HOST}\nx-date:{amz_date}\n"
    signed_headers = "content-type;host;x-date"
    canonical_request = f"{method}\n{url_path}\n{query}\n{canonical_headers}\n{signed_headers}\n{body_hash}"
    credential_scope = f"{date_stamp}/{_REGION}/{_SERVICE}/request"
    string_to_sign = f"HMAC-SHA256\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    k_date = hmac.new(sk.encode(), date_stamp.encode(), hashlib.sha256).digest()
    k_region = hmac.new(k_date, _REGION.encode(), hashlib.sha256).digest()
    k_service = hmac.new(k_region, _SERVICE.encode(), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"request", hashlib.sha256).digest()
    sig = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()
    auth = f"HMAC-SHA256 Credential={ak}/{credential_scope}, SignedHeaders={signed_headers}, Signature={sig}"
    return {"Content-Type": "application/json", "X-Date": amz_date, "Authorization": auth}

def _call(action, body, ak, sk):
    """调用即梦API"""
    qs = f"Action={action}&Version=2022-08-31"
    url = f"https://{_HOST}/?{qs}"
    b = json.dumps(body, ensure_ascii=False)
    hdrs = _sign("POST", "/", qs, b, ak, sk)
    req = urllib.request.Request(url, data=b.encode("utf-8"), headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())

def generate_image(prompt, width=1024, height=1024, req_key="jimeng_t2i_v30", timeout=60):
    """文生图 — 返回图片本地路径

    Args:
        prompt: 图片描述
        width/height: 尺寸，默认1024
        req_key: 模型版本，默认jimeng_t2i_v30(3.0)
    Returns:
        str: 本地图片路径
    """
    ak, sk = _load_keys()
    if not ak or not sk:
        raise ValueError("即梦 AK/SK 未配置，请运行 set_jimeng_keys(ak, sk)")

    # 提交任务
    body = {
        "req_key": req_key,
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": -1,
        "req_json": json.dumps({
            "return_url": True,
            "logo_info": {"add_logo": False}
        }),
    }
    resp = _call("CVSync2AsyncSubmitTask", body, ak, sk)
    if resp.get("code") != 10000:
        raise RuntimeError(f"提交失败: {resp.get('message', '未知错误')}")

    task_id = resp["data"]["task_id"]

    # 轮询结果
    for i in range(20):
        time.sleep(3)
        r = _call("CVSync2AsyncGetResult", {
            "req_key": req_key,
            "task_id": task_id,
            "req_json": json.dumps({
                "return_url": True,
                "logo_info": {"add_logo": False}
            }),
        }, ak, sk)
        data = r.get("data") or r.get("result", {})
        status = data.get("status", "?")
        imgs = data.get("image_urls") or []

        if imgs:
            # 下载图片
            import urllib.request as dl_req
            import os
            os.makedirs("output/images", exist_ok=True)
            img_url = imgs[0]
            local_path = f"output/images/jimeng_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            dl_req.urlretrieve(img_url, local_path)
            return local_path

        if status in ("failed", "error"):
            raise RuntimeError(f"生成失败: {data.get('message', '')}")

    raise TimeoutError("即梦API轮询超时")

def set_jimeng_keys(ak, sk):
    """配置即梦 AK/SK（保存到文件）"""
    _save_keys(ak, sk)
    return "✅ 即梦 Keys 已保存"


# ══════════════════════════════════════════════════════════════════
# 视频生成
# ══════════════════════════════════════════════════════════════════

def generate_video(prompt, image_url=None, req_key="jimeng_t2v_v30_1080p"):
    """文生视频/图生视频 — 返回本地视频路径

    Args:
        prompt: 视频描述
        image_url: 可选，参考图片URL（图生视频模式）
        req_key: 模型版本，默认jimeng_t2v_v30_1080p
    Returns:
        str: 本地视频路径
    """
    ak, sk = _load_keys()
    if not ak or not sk:
        raise ValueError("即梦 AK/SK 未配置")

    body = {
        "req_key": req_key,
        "prompt": prompt,
        "seed": -1,
    }
    if image_url:
        body["image_urls"] = [image_url]

    resp = _call("CVSync2AsyncSubmitTask", body, ak, sk)
    if resp.get("code") != 10000:
        raise RuntimeError(f"提交失败: {resp.get('message', '未知错误')}")

    task_id = resp["data"]["task_id"]

    # 轮询（视频生成较慢，最长等120秒）
    import os
    os.makedirs("output/videos", exist_ok=True)
    for i in range(40):
        time.sleep(3)
        r = _call("CVSync2AsyncGetResult", {
            "req_key": req_key,
            "task_id": task_id,
        }, ak, sk)
        data = r.get("data") or r.get("result", {})
        status = data.get("status", "?")
        vids = data.get("video_urls") or []

        if vids:
            import urllib.request as dl_req
            local_path = f"output/videos/jimeng_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            dl_req.urlretrieve(vids[0], local_path)
            return local_path

        if status in ("failed", "error"):
            raise RuntimeError(f"生成失败: {data.get('message', '')}")

    raise TimeoutError("即梦视频生成超时")
