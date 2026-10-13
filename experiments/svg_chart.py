"""无依赖 SVG 绘图工具（折线 / 散点 / 参考线 + 图例），用于实验报告出图。

为什么不用 matplotlib：本机 conda `stock` 环境里的 matplotlib 3.3.1 其二进制扩展是
**cp38 构建**（`ft2font.cp38-win_amd64.pyd`），而该环境是 **Python 3.14**，
`import matplotlib` 会在 `_check_versions()` 处直接 ImportError（无法加载 ft2font），
且当前无网络不便重装。本模块用纯字符串拼 SVG，零依赖、离线可生成，
浏览器 / 编辑器预览均可直接查看，也不改动用户环境。

用法：
    c = Chart(title="测试样本误差曲线")
    ax = c.axes(0, 0, title="各模型逐折 MAE", xlabel="折", ylabel="MAE(%)")
    ax.line([1, 2, 3], [1.2, 1.1, 1.3], c.color(0), label="XGBoost")
    ax.scatter(xs, ys, c.color(1))
    c.save("out.svg")
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

# ---- 主题（浅色，白底深字，保证任何环境下都清晰）----
BG = "#ffffff"
PANEL_BG = "#fcfdff"
FG = "#111827"
SUB = "#5b6472"
GRID = "#e9eef5"
AXIS = "#8b95a5"
FONT = "'Microsoft YaHei','PingFang SC','Hiragino Sans GB','Noto Sans CJK SC',sans-serif"
MONO = "'Cascadia Mono','Consolas',monospace"

PALETTE = [
    "#2563eb",  # 蓝
    "#dc2626",  # 红
    "#059669",  # 绿
    "#d97706",  # 橙
    "#7c3aed",  # 紫
    "#0891b2",  # 青
    "#db2777",  # 玫红
    "#65a30d",  # 橄榄
    "#475569",  # 灰蓝（基准）
]


def nice_ticks(lo: float, hi: float, n: int = 5) -> tuple[list[float], float]:
    """给出「好看」的刻度序列与步长。"""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo], 1.0
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = mag
    for m in (1, 1.5, 2, 2.5, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    start = math.ceil(lo / step - 1e-9) * step
    ticks, v = [], start
    while v <= hi + step * 1e-6:
        ticks.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return ticks, step


def _fmt(v: float, step: float) -> str:
    if v == 0:
        return "0"
    dec = max(0, min(4, -math.floor(math.log10(abs(step))))) if step > 0 else 2
    s = f"{v:.{dec}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


class Axes:
    """一个绘图面板。x/y 均为数据单位，内部映射到像素。"""

    def __init__(self, chart: "Chart", x0: float, y0: float, w: float, h: float,
                 title: str = "", xlabel: str = "", ylabel: str = "",
                 xlim=None, ylim=None, xticks=None, xtick_labels=None, yticks=None,
                 y_fmt=None):
        self.c = chart
        self.x0, self.y0, self.w, self.h = x0, y0, w, h  # y0 = 绘图区顶部
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.xlim, self.ylim = xlim, ylim
        self._xticks, self._xtick_labels = xticks, xtick_labels
        self._yticks, self._y_fmt = yticks, y_fmt
        self._parts: list[str] = []
        self._legend: list[tuple[str, str, str | None]] = []
        self._nline = 0

    # ---- 坐标映射 ----
    def _span(self):
        xlim = self.xlim or (0.0, 1.0)
        ylim = self.ylim or (0.0, 1.0)
        if xlim[1] <= xlim[0]:
            xlim = (xlim[0], xlim[0] + 1)
        if ylim[1] <= ylim[0]:
            ylim = (ylim[0], ylim[0] + 1)
        return xlim, ylim

    def px(self, x: float) -> float:
        (x0, x1), _ = self._span()
        return self.x0 + (x - x0) / (x1 - x0) * self.w

    def py(self, y: float) -> float:
        _, (y0, y1) = self._span()
        return self.y0 + self.h - (y - y0) / (y1 - y0) * self.h

    def color(self, i: int) -> str:
        return PALETTE[i % len(PALETTE)]

    # ---- 元素 ----
    def grid_and_ticks(self) -> "Axes":
        (x0, x1), (y0, y1) = self._span()
        yt = self._yticks if self._yticks is not None else nice_ticks(y0, y1)[0]
        ystep = (yt[1] - yt[0]) if len(yt) > 1 else (y1 - y0) / 4
        for v in yt:
            yy = self.py(v)
            self._parts.append(
                f'<line x1="{self.x0:.1f}" y1="{yy:.1f}" x2="{self.x0 + self.w:.1f}" '
                f'y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            lab = self._y_fmt(v) if self._y_fmt else _fmt(v, ystep)
            self._parts.append(
                f'<text x="{self.x0 - 8:.1f}" y="{yy + 3.5:.1f}" font-size="11" fill="{SUB}" '
                f'text-anchor="end" font-family="{MONO}">{escape(lab)}</text>')

        if self._xticks is not None:
            for i, v in enumerate(self._xticks):
                xx = self.px(v)
                lab = self._xtick_labels[i] if self._xtick_labels else str(v)
                self._parts.append(
                    f'<line x1="{xx:.1f}" y1="{self.y0:.1f}" x2="{xx:.1f}" '
                    f'y2="{self.y0 + self.h:.1f}" stroke="{GRID}" stroke-width="1"/>')
                self._parts.append(
                    f'<text x="{xx:.1f}" y="{self.y0 + self.h + 16:.1f}" font-size="11" '
                    f'fill="{SUB}" text-anchor="middle" font-family="{FONT}">{escape(lab)}</text>')
        # 边框
        self._parts.append(
            f'<rect x="{self.x0:.1f}" y="{self.y0:.1f}" width="{self.w:.1f}" '
            f'height="{self.h:.1f}" fill="none" stroke="{AXIS}" stroke-width="1"/>')
        return self

    def line(self, xs, ys, color: str, label: str | None = None, width: float = 2.0,
             dash: str | None = None, marker: bool = False) -> "Axes":
        pts = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}"
                       for x, y in zip(xs, ys) if y is not None and math.isfinite(float(y)))
        if pts:
            d = f' stroke-dasharray="{dash}"' if dash else ""
            self._parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{d}/>')
        if marker:
            for x, y in zip(xs, ys):
                if y is None or not math.isfinite(float(y)):
                    continue
                self._parts.append(
                    f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="2.6" fill="{color}"/>')
        if label:
            self._legend.append((label, color, dash))
        self._nline += 1
        return self

    def hline(self, y: float, color: str, label: str | None = None,
              dash: str = "5 4", width: float = 1.2) -> "Axes":
        yy = self.py(y)
        self._parts.append(
            f'<line x1="{self.x0:.1f}" y1="{yy:.1f}" x2="{self.x0 + self.w:.1f}" y2="{yy:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
        if label:
            self._legend.append((label, color, dash))
        return self

    def scatter(self, xs, ys, color: str, r: float = 2.2, opacity: float = 0.45,
                label: str | None = None) -> "Axes":
        for x, y in zip(xs, ys):
            if y is None or not math.isfinite(float(y)):
                continue
            self._parts.append(
                f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="{r}" '
                f'fill="{color}" fill-opacity="{opacity}"/>')
        if label:
            self._legend.append((label, color, None))
        return self

    def _render_legend(self) -> str:
        if not self._legend:
            return ""
        out, lx, ly = [], self.x0 + self.w - 8, self.y0 + 14
        for i, (lab, color, dash) in enumerate(self._legend):
            yy = ly + i * 15
            d = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<line x1="{lx - 92:.1f}" y1="{yy:.1f}" x2="{lx - 70:.1f}" y2="{yy:.1f}" '
                       f'stroke="{color}" stroke-width="2.4"{d}/>')
            out.append(f'<text x="{lx - 66:.1f}" y="{yy + 3.5:.1f}" font-size="11" fill="{FG}" '
                       f'font-family="{FONT}">{escape(str(lab))}</text>')
        return "".join(out)

    def render(self) -> str:
        out = [f'<rect x="{self.x0 - 6:.1f}" y="{self.y0 - 26:.1f}" '
               f'width="{self.w + 12:.1f}" height="{self.h + 52:.1f}" fill="{PANEL_BG}" '
               f'stroke="#eef2f7" stroke-width="1" rx="6"/>']
        out.append(f'<text x="{self.x0:.1f}" y="{self.y0 - 10:.1f}" font-size="13" '
                   f'font-weight="600" fill="{FG}" font-family="{FONT}">{escape(self.title)}</text>')
        out.extend(self._parts)
        out.append(self._render_legend())
        if self.xlabel:
            out.append(f'<text x="{self.x0 + self.w / 2:.1f}" y="{self.y0 + self.h + 34:.1f}" '
                       f'font-size="11.5" fill="{SUB}" text-anchor="middle" '
                       f'font-family="{FONT}">{escape(self.xlabel)}</text>')
        if self.ylabel:
            cy = self.y0 + self.h / 2
            out.append(f'<text x="{self.x0 - 46:.1f}" y="{cy:.1f}" font-size="11.5" fill="{SUB}" '
                       f'text-anchor="middle" font-family="{FONT}" '
                       f'transform="rotate(-90 {self.x0 - 46:.1f} {cy:.1f})">{escape(self.ylabel)}</text>')
        return "".join(out)


class Chart:
    """多面板 SVG 画布。"""

    def __init__(self, title: str = "", subtitle: str = "", ncols: int = 2, nrows: int = 2,
                 width: float = 1180, panel_h: float = 300, panel_w: float = 520,
                 left: float = 78, top: float = 92, gap_x: float = 46, gap_y: float = 62):
        self.title, self.subtitle = title, subtitle
        self.ncols, self.nrows = ncols, nrows
        self.width = width
        self.panel_w, self.panel_h = panel_w, panel_h
        self.left, self.top, self.gap_x, self.gap_y = left, top, gap_x, gap_y
        self._axes: list[Axes] = []

    def color(self, i: int) -> str:
        return PALETTE[i % len(PALETTE)]

    def axes(self, row: int, col: int, title: str = "", xlabel: str = "", ylabel: str = "",
             xlim=None, ylim=None, xticks=None, xtick_labels=None, yticks=None,
             y_fmt=None) -> Axes:
        x0 = self.left + col * (self.panel_w + self.gap_x)
        y0 = self.top + row * (self.panel_h + self.gap_y)
        ax = Axes(self, x0, y0, self.panel_w, self.panel_h, title, xlabel, ylabel,
                  xlim, ylim, xticks, xtick_labels, yticks, y_fmt)
        self._axes.append(ax)
        return ax

    def svg(self) -> str:
        height = self.top + self.nrows * (self.panel_h + self.gap_y) + 16
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" '
               f'height="{height:.0f}" viewBox="0 0 {self.width:.0f} {height:.0f}" '
               f'font-family="{FONT}">',
               f'<rect width="{self.width:.0f}" height="{height:.0f}" fill="{BG}"/>',
               f'<text x="34" y="42" font-size="21" font-weight="700" fill="{FG}">{escape(self.title)}</text>']
        if self.subtitle:
            out.append(f'<text x="34" y="65" font-size="12.5" fill="{SUB}">{escape(self.subtitle)}</text>')
        for ax in self._axes:
            ax.grid_and_ticks()
            out.append(ax.render())
        out.append("</svg>")
        return "".join(out)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.svg(), encoding="utf-8")
        return p


def html_report(title: str, subtitle: str, svg: str, sections: list[tuple[str, str]]) -> str:
    """把 SVG 与若干 HTML 片段（标题 + 表格）包成一份可直接打开的报告。"""
    body = "".join(
        f'<section><h2>{escape(t)}</h2>{c}</section>' for t, c in sections
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{escape(title)}</title>
<style>
  body {{ margin: 0; padding: 32px 32px 64px; background: #f6f8fb; color: {FG};
         font-family: {FONT}; line-height: 1.6; }}
  .wrap {{ max-width: 1240px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; }}
  .sub {{ color: {SUB}; font-size: 13.5px; margin-bottom: 22px; }}
  section {{ background: #fff; border: 1px solid #e8edf3; border-radius: 10px;
             padding: 18px 22px; margin: 18px 0; overflow-x: auto; }}
  h2 {{ font-size: 16px; margin: 0 0 12px; padding-bottom: 8px; border-bottom: 1px solid #eef2f7; }}
  table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
  th, td {{ border-bottom: 1px solid #eef2f7; padding: 6px 10px; text-align: right;
            white-space: nowrap; }}
  th {{ background: #f8fafc; text-align: right; font-weight: 600; color: {SUB}; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .svgbox {{ background: #fff; border: 1px solid #e8edf3; border-radius: 10px; padding: 8px; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
  ul {{ margin: 6px 0 0 18px; padding: 0; }} li {{ margin: 4px 0; }}
</style></head>
<body><div class="wrap">
<h1>{escape(title)}</h1>
<div class="sub">{escape(subtitle)}</div>
<div class="svgbox">{svg}</div>
{body}
</div></body></html>"""
