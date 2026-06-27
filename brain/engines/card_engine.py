"""HTML+Playwright 截图生成封面/卡片
从旧项目 html_screenshot.py 提取的独立引擎，供 Agent 工具调用。
"""

import os
import re
import html as html_mod  # 标准库，用于 HTML escape

from pathlib import Path
from playwright.sync_api import sync_playwright

# ── 常量 ──────────────────────────────────────────────────────
CANVAS_W = 1080
CANVAS_H = 1440
ACCENT = "#E94560"
DARK_BG_1 = "#1A1A2E"
DARK_BG_2 = "#16213E"
FONT = "'Microsoft YaHei', '微软雅黑', 'PingFang SC', 'Noto Sans SC', sans-serif"

OUTPUT_DIR = "output/images/"

# ── 辅助 ──────────────────────────────────────────────────────


def _safe_filename(text: str, max_len: int = 30) -> str:
    """将文本转为安全的文件名片段"""
    safe = re.sub(r"[^\w\u4e00-\u9fff \-]", "", text)[:max_len]
    return safe.strip() or "untitled"


def _screenshot(html_str: str, output_path: str, full_page: bool = False):
    """用 Playwright 对 HTML 字符串截图，不写临时文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": CANVAS_W, "height": CANVAS_H})
        page.set_content(html_str)
        page.screenshot(path=output_path, full_page=full_page)
        browser.close()
    return output_path


# ── CSS 公共片段 ──────────────────────────────────────────────

_BASE_RESET = """\
* { margin:0; padding:0; box-sizing:border-box; }
html, body { width:1080px; height:1440px; font-family:%s; -webkit-font-smoothing:antialiased; }
""" % FONT

_WATERMARK_HTML = """\
<div class="watermark">工具猫 · 小红书内容工坊</div>
"""

_WATERMARK_CSS = """\
.watermark {
  position:absolute; bottom:40px; left:0; right:0; text-align:center;
  font-size:22px; color:rgba(255,255,255,0.25); letter-spacing:2px;
}
"""


# ═══════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════


class CardGenerator:
    """HTML+Playwright 截图生成封面/卡片"""

    # ── 封面 ──────────────────────────────────────────────

    def generate_cover(self, title: str = "", subtitle: str = "",
                       tags: list = None, output_path: str = None) -> str:
        """生成小红书封面 — 深色渐变背景 + 大字居中 + 标签 + 水印"""
        tags = tags or []

        if output_path is None:
            safe = _safe_filename(title)
            output_path = os.path.join(OUTPUT_DIR, f"cover_{safe}.png")

        # 自动换行逻辑：每行最多 8 个中文字符
        title_lines = self._auto_break(title, 8)
        subtitle_lines = self._auto_break(subtitle, 16) if subtitle else []

        # 构建标签 HTML (HTML escape 防止 XSS)
        tags_html = ""
        if tags:
            tag_items = "".join(
                f'<span class="tag">#{html_mod.escape(t)}</span>' for t in tags[:5]
            )
            tags_html = f'<div class="tags">{tag_items}</div>'

        # 标题行拼接 (HTML escape 防止 XSS)
        title_html = "".join(
            f'<div class="title-line">{html_mod.escape(line)}</div>' for line in title_lines
        )
        subtitle_html = ""
        if subtitle_lines:
            sub_html = "".join(
                f'<div class="sub-line">{html_mod.escape(line)}</div>' for line in subtitle_lines
            )
            subtitle_html = f'<div class="subtitle">{sub_html}</div>'

        page_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
{_BASE_RESET}
body {{
  width:1080px; height:1440px; overflow:hidden;
  background:linear-gradient(180deg, {DARK_BG_1} 0%, {DARK_BG_2} 100%);
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  position:relative;
}}

/* 装饰光晕 */
body::before {{
  content:''; position:absolute; top:-200px; right:-200px;
  width:600px; height:600px;
  background:radial-gradient(circle, rgba(233,69,96,0.12) 0%, transparent 70%);
  border-radius:50%;
}}
body::after {{
  content:''; position:absolute; bottom:-200px; left:-200px;
  width:500px; height:500px;
  background:radial-gradient(circle, rgba(233,69,96,0.08) 0%, transparent 70%);
  border-radius:50%;
}}

.title {{
  position:relative; z-index:1;
  text-align:center; max-width:900px; padding:0 40px;
}}
.title-line {{
  font-size:82px; font-weight:700; color:#FFFFFF;
  line-height:1.25; text-shadow:0 2px 20px rgba(0,0,0,0.3);
  margin-bottom:4px;
}}
.subtitle {{
  position:relative; z-index:1;
  margin-top:36px; text-align:center;
}}
.sub-line {{
  font-size:34px; font-weight:500; color:{ACCENT};
  line-height:1.45; letter-spacing:1px;
}}

.tags {{
  position:absolute; z-index:1;
  bottom:120px; left:0; right:0; text-align:center;
  display:flex; justify-content:center; gap:20px; flex-wrap:wrap;
  padding:0 60px;
}}
.tag {{
  display:inline-block;
  font-size:24px; color:{ACCENT}; font-weight:500;
  background:rgba(233,69,96,0.12);
  border:1px solid rgba(233,69,96,0.3);
  border-radius:20px; padding:6px 18px;
  letter-spacing:1px;
}}

{_WATERMARK_CSS}

/* 底部装饰线 */
.deco-line {{
  position:absolute; bottom:100px; left:10%; right:10%;
  height:1px; background:linear-gradient(90deg,transparent,rgba(233,69,96,0.3),transparent);
}}
</style></head>
<body>
  <div class="title">{title_html}</div>
  {subtitle_html}
  {tags_html}
  <div class="deco-line"></div>
  {_WATERMARK_HTML}
</body>
</html>"""

        print(f"[🎨 HTML截图] 渲染封面…")
        _screenshot(page_html, output_path)
        print(f"[🖼️ 生图] 封面已保存: {output_path}")
        return output_path

    # ── 卡片 ──────────────────────────────────────────────

    def generate_card(self, title: str = "", subtitle: str = "",
                      bg_color: str = None, text_color: str = None,
                      accent_color: str = None, width: int = None,
                      height: int = None, output_path: str = None) -> str:
        """生成文字卡片 — 浅色背景 + 左竖条装饰 + 左对齐 + 圆角阴影"""
        w = width or CANVAS_W
        h = height or CANVAS_H
        accent = accent_color or ACCENT

        if output_path is None:
            safe = _safe_filename(title)
            output_path = os.path.join(OUTPUT_DIR, f"card_{safe}.png")

        # 分割副标题中的功能点（用 · 或 、或换行分割）
        bullet_points = []
        if subtitle:
            raw_lines = re.split(r"[·•·、\n]", subtitle)
            for ln in raw_lines:
                ln = ln.strip()
                if ln:
                    bullet_points.append(ln)

        # 如果分割后只有1个且太长，就按字符拆分
        if len(bullet_points) <= 1 and subtitle and len(subtitle) > 10:
            bullet_points = self._auto_break(subtitle, 14)

        # 标题自动换行
        title_lines = self._auto_break(title, 10)

        # 构建标题 html (HTML escape 防止 XSS)
        title_html = "".join(
            f'<div class="card-title-line">{html_mod.escape(line)}</div>' for line in title_lines
        )

        # 构建功能点列表 (HTML escape 防止 XSS)
        bullets_html = ""
        if bullet_points:
            items = "".join(
                f'<li>🔹 {html_mod.escape(pt)}</li>' for pt in bullet_points[:8]
            )
            bullets_html = f'<ul class="bullet-list">{items}</ul>'

        page_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
{_BASE_RESET}
body {{
  width:{w}px; height:{h}px; overflow:hidden;
  background:linear-gradient(135deg, #F5F5F5 0%, #FFFFFF 100%);
  display:flex; align-items:center; justify-content:center;
  position:relative;
}}

/* 左侧彩色竖条 */
body::before {{
  content:''; position:absolute; left:0; top:60px; bottom:60px;
  width:8px; background:{accent}; border-radius:0 4px 4px 0;
}}

.card {{
  width:920px; min-height:1200px;
  background:#FFFFFF;
  border-radius:24px;
  box-shadow:0 8px 40px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04);
  padding:60px 60px 50px 80px;
  position:relative; margin-left:30px;
}}

.card-title {{
  margin-bottom:30px;
  border-bottom:2px solid {accent};
  padding-bottom:20px;
}}
.card-title-line {{
  font-size:56px; font-weight:700; color:#2D2D2D;
  line-height:1.3; text-align:left;
}}

.bullet-list {{
  list-style:none; padding:0; margin:10px 0 0 0;
}}
.bullet-list li {{
  font-size:30px; line-height:1.6; color:#444;
  padding:10px 0; border-bottom:1px solid #f0f0f0;
  text-align:left;
}}
.bullet-list li:last-child {{ border-bottom:none; }}

/* 底部装饰 */
.card-footer {{
  position:absolute; bottom:40px; right:50px;
  font-size:20px; color:#ccc; letter-spacing:1px;
}}

/* 右上角装饰角标 */
.corner {{
  position:absolute; top:24px; right:24px;
  width:8px; height:8px; border-radius:50%;
  background:{accent};
}}
.corner-dot {{
  position:absolute; top:24px; right:40px;
  width:6px; height:6px; border-radius:50%;
  background:rgba(233,69,96,0.3);
}}
</style></head>
<body>
  <div class="card">
    <div class="card-title">{title_html}</div>
    {bullets_html}
    <div class="card-footer">工具猫 · 内容工坊</div>
    <div class="corner"></div>
    <div class="corner-dot"></div>
  </div>
</body>
</html>"""

        print(f"[🎨 HTML截图] 渲染卡片…")
        _screenshot(page_html, output_path)
        print(f"[🖼️ 生图] 卡片已保存: {output_path}")
        return output_path

    # ── 内部工具 ──────────────────────────────────────────

    @staticmethod
    def _auto_break(text: str, max_chars: int = 8) -> list:
        """按中文字符数自动换行"""
        lines = []
        current = ""
        for char in text:
            current += char
            if len(current) >= max_chars:
                lines.append(current)
                current = ""
        if current:
            lines.append(current)
        return lines or [""]
