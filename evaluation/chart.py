from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any
from unicodedata import east_asian_width


def _text_width(text: str, font_size: float) -> float:
    units = sum(1.0 if east_asian_width(character) in {"W", "F"} else 0.58 for character in text)
    return units * font_size


def _label_angle(max_width: float, slot: float, font_size: float) -> float:
    for angle in (0.0, 35.0, 55.0, 75.0, 90.0):
        radians = math.radians(angle)
        projected = max_width * math.cos(radians) + font_size * math.sin(radians)
        if projected <= slot * 0.88:
            return angle
    return 90.0


def _resolve_label_angle(rotation: str | float | None, max_width: float, slot: float, font_size: float) -> float:
    if rotation is None or rotation == "auto":
        return _label_angle(max_width, slot, font_size)
    angle = float(rotation)
    if not math.isfinite(angle) or not 0 <= angle <= 90:
        raise ValueError("X-axis label rotation must be between 0 and 90 degrees.")
    return angle


def box_plot_svg(
    result: dict[str, Any], *, width: int = 900, height: int = 520,
    color: str = "#36c5a2", transparent: bool = True,
    graph_opacity: float = 1.0, x_label_size: float = 11.0,
    y_label_size: float = 11.0, title_size: float = 17.0,
    x_axis_title_size: float = 12.0, y_axis_title_size: float = 12.0,
    plot_type: str = "box", title: str = "",
    x_label_rotation: str | float | None = "auto",
    x_axis_title_gap: float = 8.0, y_axis_title_gap: float = 8.0,
    show_x_axis_title: bool = True, show_y_axis_title: bool = True,
) -> str:
    if width < 240 or height < 200:
        raise ValueError("Image size must be at least 240 x 200 pixels.")
    if not math.isfinite(graph_opacity) or not 0 <= graph_opacity <= 1:
        raise ValueError("Graph opacity must be between 0 and 1.")
    font_sizes = (x_label_size, y_label_size, x_axis_title_size, y_axis_title_size, title_size)
    if any(not math.isfinite(value) or value < 6 or value > 96 for value in font_sizes):
        raise ValueError("Label and title sizes must be between 6 and 96 pixels.")
    if any(not math.isfinite(value) or value < 0 or value > 200 for value in (x_axis_title_gap, y_axis_title_gap)):
        raise ValueError("Axis title distance must be between 0 and 200 pixels.")
    if plot_type not in {"box", "violin"}:
        raise ValueError("Plot type must be box or violin.")
    rows = result["rows"]
    labels = [str(row["category"]) for row in rows]
    max_x_label_width = max((_text_width(label, x_label_size) for label in labels), default=0.0)
    y_label_width = _text_width("0.25", y_label_size)
    y_tick_zone = y_label_width + 9.0
    margin_left = max(50.0, y_tick_zone + (y_axis_title_gap + y_axis_title_size + 8.0 if show_y_axis_title else 12.0))
    y_axis_title_x = margin_left - y_tick_zone - y_axis_title_gap - y_axis_title_size * 0.35
    margin_right = max(24.0, x_label_size * 0.5)
    title_y = max(25.0, title_size + 6.0)
    margin_top = max(45.0, title_y + 20.0)
    minimum_slot = max(16.0, x_label_size * 1.35)
    canvas_width = max(float(width), margin_left + margin_right + len(rows) * minimum_slot)
    plot_width = max(1.0, canvas_width - margin_left - margin_right)
    slot = plot_width / max(1, len(rows))
    label_angle = _resolve_label_angle(x_label_rotation, max_x_label_width, slot, x_label_size)
    radians = math.radians(label_angle)
    label_height = max_x_label_width * math.sin(radians) + x_label_size * math.cos(radians)
    label_offset = x_label_size * (1.0 if label_angle == 0 else .45)
    x_axis_title_block = (x_axis_title_gap + x_axis_title_size * 1.3) if show_x_axis_title else 10.0
    margin_bottom = max(56.0, label_offset + label_height + x_axis_title_block + 14.0)
    canvas_height = max(float(height), margin_top + margin_bottom + 80.0)
    plot_height = max(1.0, canvas_height - margin_top - margin_bottom)
    plot_bottom = margin_top + plot_height
    box_width = max(8.0, min(70.0, slot * 0.55))
    background = "" if transparent else f'<rect width="{canvas_width:g}" height="{canvas_height:g}" fill="#ffffff"/>'
    marks = [f'<path class="grid" d="M{margin_left:g} {margin_top + plot_height * (1-tick):.2f}H{margin_left + plot_width:.2f}"/><text x="{margin_left - 9:.2f}" y="{margin_top + plot_height * (1-tick) + y_label_size * .35:.2f}" text-anchor="end" class="y-label">{tick:g}</text>' for tick in (0, .25, .5, .75, 1)]
    y = lambda value: margin_top + (1 - max(0.0, min(1.0, float(value)))) * plot_height
    for index, row in enumerate(rows):
        x = margin_left + index * slot + slot / 2
        label = escape(str(row["category"]))
        if row["median"] is not None:
            left, right = x - box_width / 2, x + box_width / 2
            if plot_type == "violin":
                profile = row.get("density") or [
                    [max(0.0, row["median"] - 0.025), 0.0], [row["median"], 1.0],
                    [min(1.0, row["median"] + 0.025), 0.0],
                ]
                right_points = [(x + box_width * point[1] / 2, y(point[0])) for point in profile]
                points = right_points + [(2 * x - px, py) for px, py in reversed(right_points)]
                path = " ".join(("M" if point_index == 0 else "L") + f"{px:.2f},{py:.2f}" for point_index, (px, py) in enumerate(points))
                median_y = y(row["median"])
                median_weight = min(profile, key=lambda point: abs(point[0] - row["median"]))[1]
                median_half_width = box_width * median_weight / 2
                marks.extend([
                    f'<path class="violin" d="{path} Z" fill="{escape(color)}" fill-opacity=".55" stroke="{escape(color)}"/>',
                    f'<path class="median" stroke="{escape(color)}" d="M{x-median_half_width:.2f} {median_y:.2f}H{x+median_half_width:.2f}"/>',
                ])
            else:
                low, q1, median, q3, high = (y(row[key]) for key in ("whiskerLow", "q1", "median", "q3", "whiskerHigh"))
                marks.extend([
                    f'<path class="whisker" d="M{x:.2f} {low:.2f}V{q1:.2f}M{x:.2f} {q3:.2f}V{high:.2f}M{left:.2f} {low:.2f}H{right:.2f}M{left:.2f} {high:.2f}H{right:.2f}"/>',
                    f'<rect x="{left:.2f}" y="{q3:.2f}" width="{box_width:.2f}" height="{max(1, q1-q3):.2f}" fill="{escape(color)}" fill-opacity=".55" stroke="{escape(color)}"/>',
                    f'<path class="median" stroke="{escape(color)}" d="M{left:.2f} {median:.2f}H{right:.2f}"/>',
                    *[f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="3" fill="{escape(color)}"/>' for value in row["outliers"]],
                ])
        anchor = "middle" if label_angle == 0 else "start"
        marks.append(f'<text transform="translate({x:.2f},{plot_bottom + label_offset:.2f}) rotate({label_angle:g})" text-anchor="{anchor}" class="x-label">{label}</text>')
    chart_title = escape(str(title).strip() or f'Cosine similarity by {result["groupColumn"]}')
    x_axis_title = escape(str(result["groupColumn"]))
    x_axis_title_y = plot_bottom + label_offset + label_height + x_axis_title_gap + x_axis_title_size * 0.85
    x_axis_title_svg = (
        f'<text x="{canvas_width / 2:.2f}" y="{x_axis_title_y:.2f}" text-anchor="middle" class="x-axis-title">{x_axis_title}</text>'
        if show_x_axis_title else ""
    )
    y_axis_title_svg = (
        f'<text transform="translate({y_axis_title_x:.2f},{margin_top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle" class="y-axis-title">Cosine similarity</text>'
        if show_y_axis_title else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:g}" height="{canvas_height:g}" viewBox="0 0 {canvas_width:g} {canvas_height:g}">'
        f'<style>text{{font-family:system-ui,-apple-system,sans-serif;fill:#555}}.title{{font-size:{title_size:g}px;font-weight:600}}.x-label{{font-size:{x_label_size:g}px}}.y-label{{font-size:{y_label_size:g}px}}.x-axis-title{{font-size:{x_axis_title_size:g}px}}.y-axis-title{{font-size:{y_axis_title_size:g}px}}.axis{{stroke:#777;stroke-width:1}}.grid{{stroke:#999;stroke-opacity:.2}}.whisker{{stroke:#666;stroke-width:1.5}}.median{{stroke-width:3}}</style>'
        f'{background}<g class="graph" opacity="{graph_opacity:g}"><text x="{canvas_width / 2:.2f}" y="{title_y:.2f}" text-anchor="middle" class="title">{chart_title}</text>'
        f'<path class="axis" fill="none" d="M{margin_left:g} {margin_top:g}V{plot_bottom:g}H{margin_left + plot_width:.2f}"/>'
        f'{x_axis_title_svg}{y_axis_title_svg}'
        f'{"".join(marks)}</g></svg>'
    )


def save_svg(svg: str, target: str | Path) -> None:
    path = Path(target).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
