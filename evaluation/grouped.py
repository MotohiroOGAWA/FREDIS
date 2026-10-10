from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any, Callable
from unicodedata import east_asian_width

import numpy as np
import pandas as pd

from .io import read_table
from .core import _density_profile
from .chart import _resolve_label_angle, _text_width


def _validate_items(items: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(items, list) or not items:
        raise ValueError(f"At least one {label} is required.")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each {label} must be an object.")
        item_id, name = str(item.get("id", "")).strip(), str(item.get("name", "")).strip()
        if not item_id or not name:
            raise ValueError(f"Each {label} requires a non-empty id and name.")
        normalized.append({"id": item_id, "name": name})
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label.capitalize()} ids must be unique.")
    return normalized


def _series(items: Any) -> list[dict[str, str]]:
    normalized = _validate_items(items, "series")
    source = {str(item.get("id", "")): item for item in items}
    for item in normalized:
        color = str(source[item["id"]].get("color", "#36c5a2"))
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"Invalid series color: {color}")
        try:
            int(color[1:], 16)
        except ValueError as error:
            raise ValueError(f"Invalid series color: {color}") from error
        item["color"] = color
    return normalized


def _statistics(values: pd.Series, outlier_limit: int = 250) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[np.isfinite(numeric) & numeric.between(0, 1)].sort_values()
    invalid_count = int(len(values) - len(valid))
    if valid.empty:
        return {
            "count": 0, "invalidCount": invalid_count, "min": None, "q1": None,
            "median": None, "q3": None, "max": None, "whiskerLow": None,
            "whiskerHigh": None, "outliers": [], "outlierCount": 0,
            "density": [],
        }
    q1, median, q3 = (float(valid.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    central = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    outliers = valid[~valid.index.isin(central.index)].tolist()
    if len(outliers) > outlier_limit:
        indices = np.linspace(0, len(outliers) - 1, outlier_limit, dtype=int)
        displayed = [float(outliers[index]) for index in indices]
    else:
        displayed = [float(value) for value in outliers]
    return {
        "count": int(len(valid)), "invalidCount": invalid_count,
        "min": float(valid.min()), "q1": q1, "median": median, "q3": q3,
        "max": float(valid.max()), "whiskerLow": float(central.min()),
        "whiskerHigh": float(central.max()), "outliers": displayed,
        "outlierCount": len(outliers), "density": _density_profile(valid),
    }


def compare(
    config: dict[str, Any],
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Compare one similarity result for every configured group/series pair."""
    if not isinstance(config, dict):
        raise ValueError("Grouped evaluation configuration must be an object.")
    groups = _validate_items(config.get("groups"), "group")
    series = _series(config.get("series"))
    group_ids, series_ids = {item["id"] for item in groups}, {item["id"] for item in series}
    entries = config.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("entries must be an array.")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each result entry must be an object.")
        pair = (str(entry.get("groupId", "")), str(entry.get("seriesId", "")))
        if pair[0] not in group_ids or pair[1] not in series_ids:
            raise ValueError(f"Result entry refers to an unknown group or series: {pair}")
        if pair in by_pair:
            raise ValueError(f"Only one result can be assigned to group/series pair {pair}.")
        by_pair[pair] = entry

    cells = []
    total = len(groups) * len(series)
    completed = 0
    for group in groups:
        for item in series:
            entry = by_pair.get((group["id"], item["id"]), {})
            no_data = bool(entry.get("noData", not entry.get("path")))
            cell = {
                "groupId": group["id"], "seriesId": item["id"],
                "path": str(entry.get("path", "")),
            }
            if no_data:
                cell.update(_statistics(pd.Series([], dtype=float)))
                cell["status"] = "no-data"
            else:
                path = Path(cell["path"]).expanduser().resolve()
                if path.suffix.lower() != ".mssim":
                    raise ValueError(f"Grouped box-plot input must be an .mssim file: {path}")
                if progress:
                    progress(10 + int(65 * completed / max(1, total)), f'Reading {group["name"]} / {item["name"]}…')
                frame = read_table(path)
                cell.update(_statistics(frame["cosine_similarity"]))
                cell["sourceRows"] = int(len(frame))
                cell["status"] = "ok" if cell["count"] else "no-data"
            cells.append(cell)
            completed += 1
            if progress:
                progress(10 + int(65 * completed / max(1, total)), f"Processed {completed}/{total} result files")
    return {
        "title": str(config.get("title", "Similarity comparison")).strip() or "Similarity comparison",
        "groups": groups, "series": series, "cells": cells,
    }


def grouped_box_plot_svg(
    result: dict[str, Any], *, width: int = 1000, height: int = 600,
    transparent: bool = True, background_color: str = "#ffffff",
    text_color: str = "#555555", group_gap: float = 56.0,
    series_gap: float = 6.0, box_width: float = 0.0,
    graph_opacity: float = 1.0, x_label_size: float = 12.0,
    y_label_size: float = 11.0, title_size: float = 18.0,
    x_axis_title_size: float = 12.0, y_axis_title_size: float = 12.0,
    plot_type: str = "box",
    x_label_rotation: str | float | None = "auto",
    x_axis_title_gap: float = 8.0, y_axis_title_gap: float = 8.0,
    show_x_axis_title: bool = False, show_y_axis_title: bool = True,
) -> str:
    if width < 400 or height < 300 or width > 8192 or height > 8192:
        raise ValueError("Image size must be between 400 x 300 and 8192 x 8192 pixels.")
    groups, series, cells = result["groups"], result["series"], result["cells"]
    if group_gap < 0 or series_gap < 0 or box_width < 0:
        raise ValueError("Group gap, series gap, and box width must not be negative.")
    if not math.isfinite(graph_opacity) or not 0 <= graph_opacity <= 1:
        raise ValueError("Graph opacity must be between 0 and 1.")
    font_sizes = (x_label_size, y_label_size, x_axis_title_size, y_axis_title_size, title_size)
    if any(not math.isfinite(value) or value < 6 or value > 96 for value in font_sizes):
        raise ValueError("Label and title sizes must be between 6 and 96 pixels.")
    if any(not math.isfinite(value) or value < 0 or value > 200 for value in (x_axis_title_gap, y_axis_title_gap)):
        raise ValueError("Axis title distance must be between 0 and 200 pixels.")
    if plot_type not in {"box", "violin"}:
        raise ValueError("Plot type must be box or violin.")
    cell_by_pair = {(cell["groupId"], cell["seriesId"]): cell for cell in cells}
    group_labels = [group["name"] for group in groups]
    max_x_label_width = max((_text_width(label, x_label_size) for label in group_labels), default=0.0)
    y_label_width = _text_width("0.25", y_label_size)
    y_tick_zone = y_label_width + 9.0
    margin_left = max(56.0, y_tick_zone + (y_axis_title_gap + y_axis_title_size + 8.0 if show_y_axis_title else 12.0))
    y_axis_title_x = margin_left - y_tick_zone - y_axis_title_gap - y_axis_title_size * 0.35
    margin_right = max(28.0, x_label_size * .5)
    title_y = max(27.0, title_size + 6.0)
    legend_baseline = max(58.0, title_y + 22.0)
    margin_top = max(85.0, legend_baseline + 27.0)
    minimum_group_width = max(20.0, x_label_size * 1.35, len(series) * 3 + max(0, len(series) - 1) * series_gap)
    minimum_plot_width = len(groups) * minimum_group_width + max(0, len(groups) - 1) * group_gap
    canvas_width = max(float(width), margin_left + margin_right + minimum_plot_width)
    plot_width = canvas_width - margin_left - margin_right
    if len(groups) > 1:
        maximum_group_gap = max(0.0, (plot_width - len(groups) * 20) / (len(groups) - 1))
        resolved_group_gap = min(group_gap, maximum_group_gap)
    else:
        resolved_group_gap = 0.0
    group_width = (plot_width - resolved_group_gap * (len(groups) - 1)) / len(groups)
    if len(series) > 1:
        maximum_series_gap = max(0.0, (group_width - len(series) * 3) / (len(series) - 1))
        resolved_series_gap = min(series_gap, maximum_series_gap)
    else:
        resolved_series_gap = 0.0
    maximum_box_width = max(1.0, (group_width - resolved_series_gap * (len(series) - 1)) / len(series))
    resolved_box_width = maximum_box_width if box_width == 0 else min(box_width, maximum_box_width)
    cluster_width = resolved_box_width * len(series) + resolved_series_gap * (len(series) - 1)
    label_angle = _resolve_label_angle(x_label_rotation, max_x_label_width, group_width + resolved_group_gap, x_label_size)
    radians = math.radians(label_angle)
    label_height = max_x_label_width * math.sin(radians) + x_label_size * math.cos(radians)
    label_offset = x_label_size * (1.0 if label_angle == 0 else .45)
    x_axis_title_block = (x_axis_title_gap + x_axis_title_size * 1.3) if show_x_axis_title else 10.0
    margin_bottom = max(64.0, label_offset + label_height + x_axis_title_block + 14.0)
    canvas_height = max(float(height), margin_top + margin_bottom + 100.0)
    plot_height = canvas_height - margin_top - margin_bottom
    plot_bottom = margin_top + plot_height
    y = lambda value: margin_top + (1 - max(0.0, min(1.0, float(value)))) * plot_height
    marks = []
    for tick in (0, .25, .5, .75, 1):
        tick_y = y(tick)
        marks.append(f'<path class="grid" d="M{margin_left:g} {tick_y:.2f}H{margin_left + plot_width:.2f}"/><text x="{margin_left - 9:.2f}" y="{tick_y + y_label_size * .35:.2f}" text-anchor="end" class="y-label">{tick:g}</text>')
    for group_index, group in enumerate(groups):
        group_start = margin_left + group_index * (group_width + resolved_group_gap)
        center = group_start + group_width / 2
        anchor = "middle" if label_angle == 0 else "start"
        marks.append(f'<text transform="translate({center:.2f},{plot_bottom + label_offset:.2f}) rotate({label_angle:g})" text-anchor="{anchor}" class="x-label">{escape(group["name"])}</text>')
        if group_index:
            boundary = group_start - resolved_group_gap / 2
            marks.append(f'<path class="separator" d="M{boundary:.2f} {margin_top}V{margin_top + plot_height}"/>')
        for series_index, item in enumerate(series):
            x = (
                group_start + (group_width - cluster_width) / 2
                + series_index * (resolved_box_width + resolved_series_gap)
                + resolved_box_width / 2
            )
            cell = cell_by_pair[(group["id"], item["id"])]
            color = escape(item["color"])
            if cell["median"] is None:
                marks.append(f'<path class="no-data" d="M{x-5:.2f} {y(.05)-5:.2f}l10 10m0-10l-10 10"/><text x="{x:.2f}" y="{y(.05)-10:.2f}" text-anchor="middle" class="no-data-label">No data</text>')
                continue
            left, right = x - resolved_box_width / 2, x + resolved_box_width / 2
            if plot_type == "violin":
                profile = cell.get("density") or [
                    [max(0.0, cell["median"] - 0.025), 0.0], [cell["median"], 1.0],
                    [min(1.0, cell["median"] + 0.025), 0.0],
                ]
                right_points = [(x + resolved_box_width * point[1] / 2, y(point[0])) for point in profile]
                points = right_points + [(2 * x - px, py) for px, py in reversed(right_points)]
                path = " ".join(("M" if point_index == 0 else "L") + f"{px:.2f},{py:.2f}" for point_index, (px, py) in enumerate(points))
                median_y = y(cell["median"])
                median_weight = min(profile, key=lambda point: abs(point[0] - cell["median"]))[1]
                median_half_width = resolved_box_width * median_weight / 2
                marks.extend([
                    f'<path class="violin" d="{path} Z" fill="{color}" fill-opacity=".62" stroke="{color}"/>',
                    f'<path class="median" stroke="{color}" d="M{x-median_half_width:.2f} {median_y:.2f}H{x+median_half_width:.2f}"/>',
                ])
            else:
                low, q1, median, q3, high = (y(cell[key]) for key in ("whiskerLow", "q1", "median", "q3", "whiskerHigh"))
                marks.extend([
                    f'<path class="whisker" d="M{x:.2f} {low:.2f}V{q1:.2f}M{x:.2f} {q3:.2f}V{high:.2f}M{left:.2f} {low:.2f}H{right:.2f}M{left:.2f} {high:.2f}H{right:.2f}"/>',
                    f'<rect x="{left:.2f}" y="{q3:.2f}" width="{resolved_box_width:.2f}" height="{max(1, q1-q3):.2f}" fill="{color}" fill-opacity=".62" stroke="{color}"/>',
                    f'<path class="median" stroke="{color}" d="M{left:.2f} {median:.2f}H{right:.2f}"/>',
                    *[f'<circle cx="{x:.2f}" cy="{y(value):.2f}" r="2.2" fill="{color}"/>' for value in cell["outliers"]],
                ])
    legend_parts, legend_x = [], float(margin_left)
    for item in series:
        name = item["name"]
        legend_parts.append(
            f'<rect x="{legend_x:.2f}" y="{legend_baseline - 10:.2f}" width="14" height="10" fill="{escape(item["color"])}"/>'
            f'<text x="{legend_x + 20:.2f}" y="{legend_baseline:.2f}" class="legend">{escape(name)}</text>'
        )
        # Keep legend entries close together while allowing wider CJK glyphs.
        label_width = sum(11.0 if east_asian_width(character) in {"W", "F"} else 6.5 for character in name)
        legend_x += 20.0 + label_width + 18.0
    legend = "".join(legend_parts)
    background = "" if transparent else f'<rect width="{canvas_width:g}" height="{canvas_height:g}" fill="{escape(background_color)}"/>'
    x_axis_title_y = plot_bottom + label_offset + label_height + x_axis_title_gap + x_axis_title_size * 0.85
    x_axis_title_svg = (
        f'<text x="{canvas_width / 2:.2f}" y="{x_axis_title_y:.2f}" text-anchor="middle" class="x-axis-title">Group</text>'
        if show_x_axis_title else ""
    )
    y_axis_title_svg = (
        f'<text transform="translate({y_axis_title_x:.2f},{margin_top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle" class="y-axis-title">Cosine similarity</text>'
        if show_y_axis_title else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:g}" height="{canvas_height:g}" viewBox="0 0 {canvas_width:g} {canvas_height:g}">'
        f'<style>text{{font-family:system-ui,-apple-system,sans-serif;fill:{escape(text_color)}}}.title{{font-size:{title_size:g}px;font-weight:600}}.legend{{font-size:11px}}.x-label{{font-size:{x_label_size:g}px;font-weight:600}}.y-label{{font-size:{y_label_size:g}px}}.x-axis-title{{font-size:{x_axis_title_size:g}px}}.y-axis-title{{font-size:{y_axis_title_size:g}px}}.axis{{stroke:{escape(text_color)}}}.grid{{stroke:{escape(text_color)};stroke-opacity:.2}}.separator{{stroke:{escape(text_color)};stroke-opacity:.12}}.whisker{{stroke:{escape(text_color)};stroke-width:1.4}}.median{{stroke-width:3}}.no-data{{stroke:{escape(text_color)};stroke-width:1.5;stroke-opacity:.65}}.no-data-label{{font-size:9px;fill:{escape(text_color)};fill-opacity:.75}}</style>'
        f'{background}<g class="graph" opacity="{graph_opacity:g}"><text x="{canvas_width / 2:.2f}" y="{title_y:.2f}" text-anchor="middle" class="title">{escape(result["title"])}</text>{legend}'
        f'<path class="axis" fill="none" d="M{margin_left:g} {margin_top:g}V{plot_bottom:g}H{margin_left + plot_width:.2f}"/>'
        f'{x_axis_title_svg}{y_axis_title_svg}'
        f'{"".join(marks)}</g></svg>'
    )
