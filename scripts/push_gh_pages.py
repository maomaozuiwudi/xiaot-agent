"""推 gh-pages-content 到 gh-pages 分支"""
import base64, json, subprocess, os, sys
from pathlib import Path

REPO = "maomaozuiwudi/xiaot-agent"
SRC = Path(r"E:\任务\小红书内容工坊 Agent\gh-pages-content")
REF = "refs/heads/gh-pages"

def gh(method, endpoint, **kw):
    cmd = ["gh", "api", "--method", method, endpoint]
    for k, v in kw.items():
        cmd += ["--field", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True, input=kw.get("_input"))
    if r.returncode != 0:
        print(f"❌ {endpoint}: {r.stderr[:200]}")
        return None
    return json.loads(r.stdout)

# 获取当前 gh-pages 引用
ref = gh("GET", f"repos/{REPO}/git/{REF}")
if not ref:
    print("❌ gh-pages 分支不存在")
    sys.exit(1)

current_sha = ref["object"]["sha"]
print(f"当前 gh-pages: {current_sha[:12]}")

# 获取当前 commit 的树
commit = gh("GET", f"repos/{REPO}/git/commits/{current_sha}")
base_tree_sha = commit["tree"]["sha"]

# 为每个文件创建 blob
tree_items = []
for f in sorted(SRC.rglob("*")):
    if f.is_file():
        rel = str(f.relative_to(SRC)).replace(os.sep, "/")
        content = f.read_bytes()
        b64 = base64.b64encode(content).decode()
        blob = gh("POST", f"repos/{REPO}/git/blobs",
                  content=b64, encoding="base64")
        if blob:
            tree_items.append({"path": rel, "mode": "100644",
                               "type": "blob", "sha": blob["sha"]})
            print(f"  📄 {rel}")

if not tree_items:
    print("没有文件更新")
    sys.exit(0)

# 创建新树
tree_payload = json.dumps({"base_tree": base_tree_sha, "tree": tree_items})
new_tree = gh("POST", f"repos/{REPO}/git/trees", _input=tree_payload)
if not new_tree:
    sys.exit(1)
print(f"新树: {new_tree['sha'][:12]}")

# 创建 commit
now = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
commit_payload = json.dumps({
    "message": f"小t产品首页更新 ({now})",
    "tree": new_tree["sha"],
    "parents": [current_sha],
})
new_commit = gh("POST", f"repos/{REPO}/git/commits", _input=commit_payload)
if not new_commit:
    sys.exit(1)
print(f"新commit: {new_commit['sha'][:12]}")

# 更新引用
ref_payload = json.dumps({"sha": new_commit["sha"], "force": True})
gh("PATCH", f"repos/{REPO}/git/{REF}", _input=ref_payload)

print(f"\n✅ 已推送到 https://maomaozuiwudi.github.io/xiaot-agent/")
