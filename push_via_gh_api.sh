#!/bin/bash
set -e

cd 'E:\任务\小红书内容工坊 Agent'

OWNER="maomaozuiwudi"
REPO="xiaot-agent"
BRANCH="main"
PARENT_SHA="5eacaf4fb691f1c261be6b3e63d039b999d9b7e2"
COMMIT_SHA="93966851a2a623c3c4b559a963a477cf68f6c8a3"
GIT_NAME="Game Developer"
GIT_EMAIL="dev@example.com"
COMMIT_MSG="添加 pip 包支持: pyproject.toml + xiaot 终端入口"

BASE_API="repos/$OWNER/$REPO/git"

TMPDIR=$(mktemp -d)

echo "=== Step 1: Create blob for pyproject.toml ==="
CONTENT=$(git show $COMMIT_SHA:pyproject.toml | base64 -w0)
BLOB1=$(gh api "$BASE_API/blobs" \
  --field content="$CONTENT" \
  --field encoding="base64" \
  --jq '.sha')
echo "Blob SHA (pyproject.toml): $BLOB1"

echo "=== Step 2: Create blob for xiaot_agent_entry.py ==="
CONTENT=$(git show $COMMIT_SHA:xiaot_agent_entry.py | base64 -w0)
BLOB2=$(gh api "$BASE_API/blobs" \
  --field content="$CONTENT" \
  --field encoding="base64" \
  --jq '.sha')
echo "Blob SHA (xiaot_agent_entry.py): $BLOB2"

echo "=== Step 3: Create blob for updated README.md ==="
CONTENT=$(git show $COMMIT_SHA:README.md | base64 -w0)
BLOB3=$(gh api "$BASE_API/blobs" \
  --field content="$CONTENT" \
  --field encoding="base64" \
  --jq '.sha')
echo "Blob SHA (README.md): $BLOB3"

echo "=== Step 4: Create tree ==="
cat > "$TMPDIR/tree_payload.json" <<- PAYLOAD
{
  "base_tree": "24cce8bf978171d7650fe820a9ca57bacb158e21",
  "tree": [
    {"path": "pyproject.toml", "mode": "100644", "type": "blob", "sha": "$BLOB1"},
    {"path": "xiaot_agent_entry.py", "mode": "100644", "type": "blob", "sha": "$BLOB2"},
    {"path": "README.md", "mode": "100644", "type": "blob", "sha": "$BLOB3"}
  ]
}
PAYLOAD

NEW_TREE=$(gh api "$BASE_API/trees" \
  --input "$TMPDIR/tree_payload.json" \
  --jq '.sha')
echo "New tree SHA: $NEW_TREE"

echo "=== Step 5: Create commit ==="
cat > "$TMPDIR/commit_payload.json" <<- PAYLOAD
{
  "message": "$COMMIT_MSG",
  "tree": "$NEW_TREE",
  "parents": ["$PARENT_SHA"],
  "author": {"name": "$GIT_NAME", "email": "$GIT_EMAIL"},
  "committer": {"name": "$GIT_NAME", "email": "$GIT_EMAIL"}
}
PAYLOAD

NEW_COMMIT=$(gh api "$BASE_API/commits" \
  --input "$TMPDIR/commit_payload.json" \
  --jq '.sha')
echo "New commit SHA: $NEW_COMMIT"

echo "=== Step 6: Update ref ==="
gh api "$BASE_API/refs/heads/$BRANCH" \
  --method PATCH \
  --field sha="$NEW_COMMIT" \
  --field force=true \
  --jq '.ref'
echo "Ref updated successfully!"

rm -rf "$TMPDIR"

echo ""
echo "=== DONE ==="
echo "New commit: $NEW_COMMIT"
echo "View at: https://github.com/$OWNER/$REPO/commit/$NEW_COMMIT"
