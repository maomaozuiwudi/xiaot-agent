"""
RAG 引擎 — 纯本地，轻量，零外部向量数据库依赖

检索策略（四层递进）：
1. 关键词精确匹配
2. 标签/分类匹配（YAML 头部元数据）
3. 全文子串匹配
4. TF-IDF 语义相似度
"""
import os
import re
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config_loader import get, resolve_path


@dataclass
class DocFragment:
    """文档片段"""
    source: str          # 源文件路径
    title: str           # 文档标题（第一行或文件名）
    category: str        # 分类（风格/规则/避坑/历史/偏好/知识）
    content: str         # 匹配片段
    tags: list           # 标签
    score: float = 0.0   # 匹配分数


class RAGEngine:
    """RAG 引擎 — 四层递进检索"""

    def __init__(self, sources_dir: str = None, shared_repo: str = "", shared_enabled: bool = False):
        if sources_dir is None:
            sources_dir = resolve_path(get("knowledge.rag.sources", [])[0]
                                       if get("knowledge.rag.sources") else "skills/xhs-content-factory/references/")
        self.sources_dir = Path(sources_dir)
        self.shared_repo = shared_repo or get("knowledge.shared.repo", "")
        self.shared_enabled = shared_enabled or get("knowledge.shared.enabled", False)
        self.shared_dir = Path(resolve_path("data/shared/knowledge/")) if self.shared_repo else None
        self._index = []            # [(tags, content, path, title), ...]
        self._stopwords = {"的", "了", "是", "在", "有", "和", "就", "不", "人", "都",
                           "一", "个", "上", "也", "很", "到", "说", "要", "去", "你",
                           "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
                           "们", "那", "些", "能", "下", "过", "出", "来", "让", "对"}
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._sync_shared_repo()
        self._build_index()

    # ── 索引构建 ──



    def _build_index(self):
        """扫描参考文献目录，建立索引"""
        if not self.sources_dir.exists():
            print(f"[RAG] 参考库目录不存在: {self.sources_dir}")
            return

        md_files = list(self.sources_dir.rglob("*.md")) + list(self.sources_dir.rglob("*.yaml"))
        if self.shared_dir and self.shared_dir.exists():
            md_files += list(self.shared_dir.rglob("*.md")) + list(self.shared_dir.rglob("*.yaml"))
        for fpath in md_files:
            try:
                content = fpath.read_text("utf-8", errors="ignore")
                # 提取标题
                title = fpath.stem
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("# ") or line.startswith("## "):
                        title = line.lstrip("#").strip()
                        break

                # 提取标签（YAML frontmatter）
                tags = []
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1]
                        for fm_line in frontmatter.split("\n"):
                            fm_line = fm_line.strip()
                            if fm_line.startswith("tags:"):
                                raw = fm_line[5:].strip()
                                tags = [t.strip().strip("[]'\"") for t in raw.split(",") if t.strip()]
                                break

                # 分类（按上级目录名或来源）
                try:
                    rel_path = fpath.relative_to(self.sources_dir)
                    category = str(rel_path.parent) if rel_path.parent != Path(".") else "通用"
                except ValueError:
                    # 来自共享库的文件
                    if self.shared_dir:
                        try:
                            rel_path = fpath.relative_to(self.shared_dir)
                            category = f"共享/{str(rel_path.parent) if rel_path.parent != Path('.') else '通用'}"
                        except ValueError:
                            category = "共享/通用"
                    else:
                        category = "通用"

                self._index.append({
                    "path": str(fpath),
                    "title": title,
                    "category": category,
                    "tags": tags,
                    "content": content,
                    # 按段落分割方便匹配
                    "paragraphs": [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip() and len(p.strip()) > 20],
                })
            except Exception:
                pass

        # 构建 TF-IDF 索引（如果文档够多）
        if len(self._index) >= 3:
            self._build_tfidf()

        print(f"[RAG] 索引完成: {len(self._index)} 个文档")

    # ── GitHub 共享知识库 ──

    def _sync_shared_repo(self):
        """从 GitHub 同步共享知识库到 data/shared/"""
        if not self.shared_repo:
            return
        try:
            self.shared_dir.mkdir(parents=True, exist_ok=True)
            git_dir = self.shared_dir / ".git"
            if git_dir.exists():
                import subprocess
                result = subprocess.run(
                    ["git", "-C", str(self.shared_dir), "pull"],
                    capture_output=True, text=True, timeout=30
                )
                print(f"[RAG] 共享知识库已更新: {result.stdout.strip()[:100]}")
            else:
                import subprocess
                result = subprocess.run(
                    ["gh", "repo", "clone", self.shared_repo, str(self.shared_dir)],
                    capture_output=True, text=True, timeout=60
                )
                print(f"[RAG] 共享知识库已克隆: {result.stdout.strip()[:100]}")
        except Exception as e:
            print(f"[RAG] 共享知识库同步失败 (不影响本地使用): {e}")

    def _contribute_to_shared(self, title: str, content: str, category: str, tags: list):
        """直接写入本地共享库并推送 GitHub（无需审核）"""
        if not self.shared_repo or not self.shared_dir:
            return
        try:
            safe_name = re.sub(r'[\\/:*?"<>|#\s]', '_', title)[:40]
            safe_cat = re.sub(r'[\\/:*?"<>|#\s]', '_', str(category or "general"))[:20]

            # 写入文件到共享目录
            cat_dir = self.shared_dir / safe_cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            filepath = cat_dir / f"{safe_name}.md"
            tag_str = ", ".join(tags) if tags else ""
            filepath.write_text(
                f"---\ntitle: {title}\ntags: [{tag_str}]\n---\n\n{content}",
                encoding="utf-8"
            )

            # git add -> commit -> push（直推，不经过审核）
            import subprocess
            subprocess.run(["git", "-C", str(self.shared_dir), "add", "."],
                           capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(self.shared_dir), "commit",
                           "-m", f"共享: {title[:60]}"],
                           capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(self.shared_dir), "push"],
                           capture_output=True, timeout=30)
            print(f"[RAG] 知识已共享并推送: {filepath.name}")
        except Exception as e:
            print(f"[RAG] 共享提交异常 (不影响本地使用): {e}")

    def _build_tfidf(self):
        """构建 TF-IDF 向量索引"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            documents = []
            self._doc_map = []
            for doc in self._index:
                for para in doc["paragraphs"]:
                    documents.append(para)
                    self._doc_map.append((doc["path"], para[:100]))
            if documents:
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words=list(self._stopwords),
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_df=0.8,
                    min_df=1,
                )
                self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(documents)
        except ImportError:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None

    # ── 四层检索 ──

    def query(self, question: str, top_k: int = 3) -> List[DocFragment]:
        """
        检索最相关的文档片段
        返回排序后的 DocFragment 列表
        """
        if not self._index:
            return []

        # 分词
        keywords = self._tokenize(question)

        results = []
        seen = set()

        # 第一层：关键词精确匹配
        for doc in self._index:
            score = 0
            hit_keywords = []
            content_lower = doc["content"].lower()
            for kw in keywords:
                if kw in content_lower:
                    score += 1
                    hit_keywords.append(kw)
            if score > 0:
                # 匹配上的段落
                matched_paras = self._find_best_paragraph(doc, keywords, question)
                if matched_paras not in seen:
                    seen.add(matched_paras)
                    results.append(DocFragment(
                        source=doc["path"],
                        title=doc["title"],
                        category=doc["category"],
                        content=matched_paras,
                        tags=doc["tags"],
                        score=score / len(keywords),
                    ))

        # 第二层：标签匹配
        if len(results) < top_k:
            question_lower = question.lower()
            for doc in self._index:
                if any(t.lower() in question_lower or question_lower in t.lower()
                       for t in doc["tags"]):
                    key = doc["path"]
                    if key not in seen:
                        seen.add(key)
                        results.append(DocFragment(
                            source=doc["path"],
                            title=doc["title"],
                            category=doc["category"],
                            content=doc["paragraphs"][0][:500] if doc["paragraphs"] else doc["content"][:500],
                            tags=doc["tags"],
                            score=0.6,
                        ))

        # 第三层：全文子串匹配
        if len(results) < top_k:
            for doc in self._index:
                for kw in keywords:
                    if len(kw) >= 2:
                        for para in doc["paragraphs"]:
                            if kw in para.lower():
                                key = f"{doc['path']}:{para[:50]}"
                                if key not in seen:
                                    seen.add(key)
                                    results.append(DocFragment(
                                        source=doc["path"],
                                        title=doc["title"],
                                        category=doc["category"],
                                        content=para[:600],
                                        tags=doc["tags"],
                                        score=0.3,
                                    ))
                                    break

        # 第四层：TF-IDF 语义相似度
        if len(results) < top_k and self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            try:
                query_vec = self._tfidf_vectorizer.transform([question])
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
                top_indices = similarities.argsort()[-top_k * 2:][::-1]
                for idx in top_indices:
                    if similarities[idx] < 0.05:
                        continue
                    doc_path, para_preview = self._doc_map[idx]
                    key = f"{doc_path}:{para_preview}"
                    if key not in seen:
                        seen.add(key)
                        results.append(DocFragment(
                            source=doc_path,
                            title=Path(doc_path).stem,
                            category="",
                            content=self._find_content_by_preview(doc_path, para_preview),
                            tags=[],
                            score=similarities[idx],
                        ))
            except Exception:
                pass

        # 排序去重，取 top_k
        results.sort(key=lambda x: -x.score)
        return results[:top_k]

    # ── 辅助方法 ──

    def _tokenize(self, text: str) -> list:
        """简单中文分词"""
        text = text.lower()
        # 去标点，按空格和常见分隔符切
        text = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
        tokens = []
        for t in text.split():
            t = t.strip()
            if not t or t in self._stopwords:
                continue
            tokens.append(t)
            # 对中文单字，尝试组合相邻的双字词
            if re.match(r'^[\u4e00-\u9fff]+$', t) and len(t) >= 2:
                for i in range(len(t) - 1):
                    bigram = t[i:i+2]
                    if bigram not in self._stopwords:
                        tokens.append(bigram)
        return list(set(tokens))

    def _find_best_paragraph(self, doc: dict, keywords: list, question: str) -> str:
        """找到最匹配关键词的段落"""
        best_para = ""
        best_score = 0
        for para in doc["paragraphs"]:
            para_lower = para.lower()
            score = sum(1 for kw in keywords if kw in para_lower)
            # 额外加分：段落开头（标题附近）
            if para.startswith("#") or para.startswith("##"):
                score *= 1.3
            if score > best_score:
                best_score = score
                best_para = para
        if best_para:
            return best_para[:800]  # 截断，不要太大
        return doc["content"][:500]

    def _find_content_by_preview(self, doc_path: str, preview: str) -> str:
        """根据预览文本查找完整段落"""
        for doc in self._index:
            if doc["path"] == doc_path:
                for para in doc["paragraphs"]:
                    if preview in para:
                        return para[:600]
                return doc["content"][:500]
        return preview

    # ── 动态注入（不写文件） ──

    def ingest(self, title: str, content: str, category: str = "xhs_search",
               tags: list = None, source: str = "xhs_search") -> bool:
        """
        将内容直接注入 RAG 内存索引（不写文件）。

        Args:
            title: 文档标题（搜索关键词）
            content: Markdown 格式的完整内容
            category: 分类（默认 xhs_search）
            tags: 标签列表
            source: 来源标识（默认 xhs_search）

        Returns:
            bool: 是否成功
        """
        if not content or not content.strip():
            return False

        try:
            doc = {
                "path": f"{source}://{title}",
                "title": title,
                "category": category,
                "tags": tags or [],
                "content": content,
                "paragraphs": [p.strip() for p in re.split(r'\n\s*\n', content)
                               if p.strip() and len(p.strip()) > 20],
            }
            self._index.append(doc)

            # 更新 TF-IDF 矩阵
            if self._tfidf_vectorizer is not None and doc["paragraphs"]:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    new_paras = doc["paragraphs"]
                    new_vec = self._tfidf_vectorizer.transform(new_paras)
                    from scipy.sparse import vstack
                    self._tfidf_matrix = vstack([self._tfidf_matrix, new_vec])
                    # 更新 doc_map
                    if not hasattr(self, '_doc_map'):
                        self._doc_map = []
                    for para in new_paras:
                        self._doc_map.append((doc["path"], para[:100]))
                except Exception:
                    # TF-IDF 更新失败不影响主流程
                    pass

            return True
        except Exception:
            return False

    def ingest_and_persist(self, title: str, content: str, category: str = "user_prefs",
                           tags: list = None, source: str = "user") -> bool:
        """注入内存 + 同时写入参考库文件（持久化）"""
        ok = self.ingest(title, content, category, tags, source)
        if not ok:
            return False
        try:
            safe_name = re.sub(r'[\\/:*?"<>|#\s]', '_', title)[:40]
            safe_src = re.sub(r'[\\/:*?"<>|#\s]', '_', source)[:10]
            filepath = self.sources_dir / f"user_{safe_src}_{safe_name}.md"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
        except Exception:
            pass
        # 如果共享模式开启，自动提交到共享库
        if self.shared_enabled and self.shared_repo:
            self._contribute_to_shared(title, content, category, tags)
        return True

    def refresh(self):
        """重新索引（当参考库文件变化时）"""
        self._index = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._build_index()
