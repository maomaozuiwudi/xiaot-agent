#!/usr/bin/env python3
"""
创建 gh-pages 孤儿分支并推送初始内容
通过 GitHub API (gh CLI) 推送，无需本地 git push
"""
import json
import subprocess
import base64
import os
import sys

OWNER = "maomaozuiwudi"
REPO = "xiaot-agent"
BRANCH = "gh-pages"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT_DIR = os.path.join(PROJECT_ROOT, "gh-pages-content")


def gh_field(endpoint, fields, jq=None, method="POST"):
    """Run gh CLI with --field arguments (reliable approach)"""
    cmd = ["gh", "api", endpoint, "--method", method]
    for key, value in fields.items():
        cmd.extend(["--field", f"{key}={value}"])
    if jq:
        cmd.extend(["--jq", jq])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gh error ({endpoint}): {result.stderr[:300]}")
        return None
    return result.stdout.strip()


def main():
    print(f"=== 创建 gh-pages 孤儿分支: {OWNER}/{REPO} ===\n")
    
    # ---- Scan files from gh-pages-content ----
    file_entries = []
    for root, dirs, files in os.walk(CONTENT_DIR):
        for fname in sorted(files):
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, CONTENT_DIR).replace("\\", "/")
            
            with open(full_path, "rb") as f:
                raw = f.read()
            b64_content = base64.b64encode(raw).decode()
            
            print(f"  Creating blob: {rel_path} ({len(raw)} bytes)")
            blob_sha = gh_field(
                f"repos/{OWNER}/{REPO}/git/blobs",
                {"content": b64_content, "encoding": "base64"},
                jq=".sha"
            )
            if not blob_sha:
                print(f"  FAILED to create blob for {rel_path}")
                return
            
            file_entries.append({
                "path": rel_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha
            })
            print(f"    → blob: {blob_sha[:12]}...")
    
    if not file_entries:
        print("Error: No files found in gh-pages-content/")
        return
    
    # ---- Create tree ----
    print(f"\n=== Creating tree with {len(file_entries)} files ===")
    # For tree creation, we need to use JSON input
    tree_payload = json.dumps({"tree": file_entries})
    result = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{REPO}/git/trees",
         "--input", "-", "--jq", ".sha"],
        capture_output=True, text=True, input=tree_payload
    )
    if result.returncode != 0:
        print(f"Error creating tree: {result.stderr[:300]}")
        return
    tree_sha = result.stdout.strip()
    print(f"Tree SHA: {tree_sha}")
    
    # ---- Create commit ----
    print("\n=== Creating commit ===")
    commit_payload = json.dumps({
        "message": "初始化 gh-pages 分支 - 静态仪表盘页面\n\n通过 GitHub Pages 在手机上访问仪表盘，无需内网穿透。",
        "tree": tree_sha,
        "parents": [],
        "author": {"name": "Game Developer", "email": "dev@example.com"},
        "committer": {"name": "Game Developer", "email": "dev@example.com"}
    })
    result = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{REPO}/git/commits",
         "--input", "-", "--jq", ".sha"],
        capture_output=True, text=True, input=commit_payload
    )
    if result.returncode != 0:
        print(f"Error creating commit: {result.stderr[:300]}")
        return
    commit_sha = result.stdout.strip()
    print(f"Commit SHA: {commit_sha}")
    
    # ---- Create or update ref ----
    print(f"\n=== Creating/updating ref: refs/heads/{BRANCH} ===")
    
    # Check if ref exists
    check = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
         "--jq", ".ref"],
        capture_output=True, text=True
    )
    ref_exists = check.returncode == 0
    
    if ref_exists:
        print("  Ref exists, updating (force)...")
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
             "--method", "PATCH",
             "--field", f"sha={commit_sha}",
             "--field", "force=true",
             "--jq", ".ref"],
            capture_output=True, text=True
        )
    else:
        print("  Creating new ref...")
        ref_payload = json.dumps({
            "ref": f"refs/heads/{BRANCH}",
            "sha": commit_sha
        })
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/git/refs",
             "--input", "-", "--jq", ".ref"],
            capture_output=True, text=True, input=ref_payload
        )
    
    if result.returncode != 0:
        print(f"  Failed: {result.stderr[:200]}")
        return
    print(f"  ✅ Ref: {result.stdout.strip()}")
    
    # ---- Try to configure GitHub Pages ----
    print("\n=== GitHub Pages configuration ===")
    pages_check = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{REPO}/pages", "--jq", ".source.branch"],
        capture_output=True, text=True
    )
    if pages_check.returncode == 0 and pages_check.stdout.strip():
        print(f"  ✅ GitHub Pages already configured: branch={pages_check.stdout.strip()}")
    else:
        print("  Attempting to enable GitHub Pages from gh-pages branch...")
        pages_payload = json.dumps({
            "source": {"branch": BRANCH, "path": "/"}
        })
        result = subprocess.run(
            ["gh", "api", f"repos/{OWNER}/{REPO}/pages",
             "--input", "-", "--jq", ".source.branch"],
            capture_output=True, text=True, input=pages_payload
        )
        if result.returncode == 0:
            print(f"  ✅ GitHub Pages enabled: branch={result.stdout.strip()}")
        else:
            print(f"  ℹ️  Manual config needed: {result.stderr[:200]}")
    
    print(f"\n{'='*50}")
    print(f"✅ gh-pages 分支已创建!")
    print(f"Commit: {commit_sha}")
    print(f"\n文件列表:")
    for e in file_entries:
        print(f"  📄 {e['path']}")
    print(f"\n访问地址:")
    print(f"  https://{OWNER}.github.io/{REPO}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
