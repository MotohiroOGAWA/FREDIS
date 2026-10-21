from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Sequence

from .chart import box_plot_svg, save_svg
from .core import EvaluationRequest, inspect_dataset, summarize
from .grouped import compare, grouped_box_plot_svg


PROGRESS_PREFIX = "FREDIS_PROGRESS "


def _json_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array.")
    return tuple(str(item) for item in parsed)


def _bins(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _smarts_queries(value: str | None, plain: Sequence[str]) -> tuple[tuple[str, str], ...]:
    items = [] if value is None else json.loads(value)
    if not isinstance(items, list):
        raise ValueError("--smarts-json must be a JSON array.")
    items = [*items, *plain]
    normalized = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            label = smarts = item
        elif isinstance(item, dict):
            label, smarts = item.get("label", item.get("name", "")), item.get("smarts", "")
        elif isinstance(item, list) and len(item) == 2:
            label, smarts = item
        else:
            raise ValueError(
                f"SMARTS query #{index} must be a string, [label, smarts], or an object."
            )
        normalized.append((str(label).strip(), str(smarts).strip()))
    return tuple(normalized)


def _grouped_config(args: argparse.Namespace) -> dict:
    if args.request_json:
        value = json.loads(args.request_json)
    else:
        with open(args.config, encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Grouped evaluation configuration must be a JSON object.")
    if value.get("schema") == "fredis.evaluation.config":
        if value.get("schemaVersion") != 1 or value.get("kind") != "grouped":
            raise ValueError("Expected a version 1 grouped evaluation configuration.")
        value = value.get("config")
        if not isinstance(value, dict):
            raise ValueError("Grouped evaluation configuration payload is missing.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description="Group and summarize FREDIS result tables.")
    commands = parser.add_subparsers(dest="command", required=True)
    presentation = commands.add_parser('filtering-view', help='Render or export cached filtering counts.')
    presentation.add_argument('--request-json', required=True)
    destination = presentation.add_mutually_exclusive_group()
    destination.add_argument('--output-dir')
    destination.add_argument('--output-file')
    presentation.add_argument('--format', choices=['tsv','png','svg'])
    filtering = commands.add_parser("filtering", help="Audit trainable SpecID retention.")
    filtering.add_argument('--request-json', required=True)
    filtering.add_argument('--output-dir')
    filtering.add_argument('--inspect', action='store_true')
    inspect = commands.add_parser("inspect", help="Inspect input columns and inferred types.")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--metadata")
    inspect.add_argument("--join-column", help="Join column in the result table.")
    inspect.add_argument("--metadata-join-column", help="Join column in the metadata source.")
    summary = commands.add_parser("summarize", help="Create grouped cosine-similarity box-plot data.")
    summary.add_argument("--input", required=True)
    summary.add_argument("--metadata")
    summary.add_argument("--join-column", help="Join column in the result table.")
    summary.add_argument("--metadata-join-column", help="Join column in the metadata source.")
    summary.add_argument("--group-column", required=True)
    summary.add_argument("--mode", choices=("auto", "categorical", "numeric"), default="auto")
    summary.add_argument(
        "--transform",
        choices=("none", "collision-energy", "chemical", "smarts"),
        default="none",
    )
    summary.add_argument("--precursor-mz-column", default="PrecursorMZ")
    summary.add_argument("--instrument-column")
    summary.add_argument("--smiles-column", default="SMILES")
    summary.add_argument("--chemical-descriptor", default="HeavyAtomCount")
    summary.add_argument(
        "--smarts", action="append", default=[],
        help="SMARTS query to evaluate. Repeat for multiple independently matching groups.",
    )
    summary.add_argument(
        "--smarts-json",
        help='JSON array of {"label": "...", "smarts": "..."} queries.',
    )
    summary.add_argument("--bins", default="", help="Comma-separated boundaries, e.g. 0,10,20.")
    summary.add_argument("--include", help="JSON array of category/range labels to include.")
    summary.add_argument("--order", help="JSON array defining output order.")
    summary.add_argument("--width", type=int, default=900)
    summary.add_argument("--height", type=int, default=520)
    summary.add_argument("--color", default="#36c5a2")
    summary.add_argument("--graph-opacity", type=float, default=1.0)
    summary.add_argument("--x-label-size", type=float, default=11.0)
    summary.add_argument("--y-label-size", type=float, default=11.0)
    summary.add_argument("--x-axis-title-size", type=float, default=12.0)
    summary.add_argument("--y-axis-title-size", type=float, default=12.0)
    summary.add_argument("--x-axis-title-gap", type=float, default=8.0)
    summary.add_argument("--y-axis-title-gap", type=float, default=8.0)
    summary.add_argument("--hide-x-axis-title", action="store_true")
    summary.add_argument("--hide-y-axis-title", action="store_true")
    summary.add_argument("--x-label-rotation", default="auto", help="auto or an angle in degrees (0-90).")
    summary.add_argument("--title-size", type=float, default=17.0)
    summary.add_argument("--title", default="")
    summary.add_argument("--plot-type", choices=("box", "violin"), default="box")
    summary.add_argument("--opaque", action="store_true", help="Use a white background instead of transparency.")
    summary.add_argument("--output-image", help="Optional SVG output path.")
    summary.add_argument("--output-json", help="Optional JSON result path.")
    grouped = commands.add_parser("compare", help="Compare multiple .mssim files by group and series.")
    source = grouped.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Grouped evaluation JSON configuration path.")
    source.add_argument("--request-json", help="Grouped evaluation configuration as JSON.")
    grouped.add_argument("--width", type=int)
    grouped.add_argument("--height", type=int)
    grouped.add_argument("--opaque", action="store_true", help="Use a solid background instead of transparency.")
    grouped.add_argument("--background-color")
    grouped.add_argument("--text-color")
    grouped.add_argument("--graph-opacity", type=float)
    grouped.add_argument("--x-label-size", type=float)
    grouped.add_argument("--y-label-size", type=float)
    grouped.add_argument("--x-axis-title-size", type=float)
    grouped.add_argument("--y-axis-title-size", type=float)
    grouped.add_argument("--x-axis-title-gap", type=float)
    grouped.add_argument("--y-axis-title-gap", type=float)
    grouped.add_argument("--hide-x-axis-title", action="store_true")
    grouped.add_argument("--hide-y-axis-title", action="store_true")
    grouped.add_argument("--x-label-rotation", default=None, help="auto or an angle in degrees (0-90).")
    grouped.add_argument("--title-size", type=float)
    grouped.add_argument("--plot-type", choices=("box", "violin"))
    grouped.add_argument("--output-image", help="Optional SVG output path.")
    grouped.add_argument("--output-json", help="Optional JSON result path.")
    return parser


def execute(
    args: argparse.Namespace,
    progress: Callable[[int, str], None] | None = None,
) -> dict:
    if args.command == 'filtering-view':
        from .filtering import present, enrich_metadata
        payload = json.load(sys.stdin) if args.request_json == '-' else json.loads(args.request_json)
        enriched = enrich_metadata(payload['result'], payload.get('view') or {}, payload.get('metadataConfig'))
        rendered = present(enriched, payload.get('view'), args.output_dir, [args.format] if args.format else [], output_file=args.output_file)
        if enriched is not payload['result']:
            rendered['updatedResult'] = enriched
        return rendered
    if args.command == "filtering":
        from .filtering import evaluate, inspect_metadata
        config = json.loads(args.request_json)
        return inspect_metadata(config) if args.inspect else evaluate(config, args.output_dir)
    if args.command == "inspect":
        if progress:
            progress(10, "Reading dataset columns…")
        result = inspect_dataset(
            args.input, args.metadata, args.join_column, args.metadata_join_column
        )
        if progress:
            progress(100, "Columns loaded")
        return result
    if args.command == "compare":
        config = _grouped_config(args)
        if progress:
            progress(5, "Preparing grouped evaluation…")
        result = compare(config, progress=progress)
        if progress:
            progress(85, "Rendering grouped SVG…")
        result["svg"] = grouped_box_plot_svg(
            result,
            width=args.width or int(config.get("width", 1000)),
            height=args.height or int(config.get("height", 600)),
            transparent=not args.opaque and bool(config.get("transparent", True)),
            background_color=args.background_color or str(config.get("backgroundColor", "#ffffff")),
            text_color=args.text_color or str(config.get("textColor", "#555555")),
            group_gap=float(config.get("groupGap", 56)),
            series_gap=float(config.get("seriesGap", 6)),
            box_width=float(config.get("boxWidth", 0)),
            graph_opacity=args.graph_opacity if args.graph_opacity is not None else float(config.get("graphOpacity", 1)),
            x_label_size=args.x_label_size if args.x_label_size is not None else float(config.get("xLabelSize", 12)),
            y_label_size=args.y_label_size if args.y_label_size is not None else float(config.get("yLabelSize", 11)),
            x_axis_title_size=args.x_axis_title_size if args.x_axis_title_size is not None else float(config.get("xAxisTitleSize", config.get("xLabelSize", 12))),
            y_axis_title_size=args.y_axis_title_size if args.y_axis_title_size is not None else float(config.get("yAxisTitleSize", config.get("yLabelSize", 12))),
            x_axis_title_gap=args.x_axis_title_gap if args.x_axis_title_gap is not None else float(config.get("xAxisTitleGap", 8)),
            y_axis_title_gap=args.y_axis_title_gap if args.y_axis_title_gap is not None else float(config.get("yAxisTitleGap", 8)),
            show_x_axis_title=not args.hide_x_axis_title and bool(config.get("showXAxisTitle", False)),
            show_y_axis_title=not args.hide_y_axis_title and bool(config.get("showYAxisTitle", True)),
            x_label_rotation=args.x_label_rotation if args.x_label_rotation is not None else config.get("xLabelRotation", "auto"),
            title_size=args.title_size if args.title_size is not None else float(config.get("titleSize", 18)),
            plot_type=args.plot_type or str(config.get("plotType", "box")),
        )
        if args.output_image:
            if progress:
                progress(95, "Saving SVG…")
            save_svg(result["svg"], args.output_image)
            result["imagePath"] = args.output_image
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as stream:
                json.dump({key: value for key, value in result.items() if key != "svg"}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        if progress:
            progress(100, "Grouped evaluation complete")
        return result
    request = EvaluationRequest(
        input_path=args.input, group_column=args.group_column,
        metadata=args.metadata, join_column=args.join_column,
        metadata_join_column=args.metadata_join_column, mode=args.mode,
        bins=_bins(args.bins), include=_json_list(args.include), order=_json_list(args.order) or (),
        transform=args.transform, precursor_mz_column=args.precursor_mz_column,
        instrument_column=args.instrument_column, smiles_column=args.smiles_column,
        chemical_descriptor=args.chemical_descriptor,
        smarts_queries=_smarts_queries(args.smarts_json, args.smarts),
    )
    result = summarize(request, progress=progress)
    if progress:
        progress(85, "Rendering evaluation SVG…")
    result["svg"] = box_plot_svg(
        result, width=args.width, height=args.height, color=args.color,
        transparent=not args.opaque, graph_opacity=args.graph_opacity,
        x_label_size=args.x_label_size, y_label_size=args.y_label_size,
        x_axis_title_size=args.x_axis_title_size,
        y_axis_title_size=args.y_axis_title_size,
        x_axis_title_gap=args.x_axis_title_gap,
        y_axis_title_gap=args.y_axis_title_gap,
        show_x_axis_title=not args.hide_x_axis_title,
        show_y_axis_title=not args.hide_y_axis_title,
        x_label_rotation=args.x_label_rotation,
        title_size=args.title_size,
        plot_type=args.plot_type, title=args.title,
    )
    if args.output_image:
        if progress:
            progress(95, "Saving SVG…")
        save_svg(result["svg"], args.output_image)
        result["imagePath"] = args.output_image
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as stream:
            json.dump({key: value for key, value in result.items() if key != "svg"}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if progress:
        progress(100, "Evaluation complete")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    try:
        def report_progress(percent: int, message: str) -> None:
            payload = json.dumps({"percent": percent, "message": message}, ensure_ascii=False)
            print(PROGRESS_PREFIX + payload, file=sys.stderr, flush=True)

        result = execute(build_parser().parse_args(argv), progress=report_progress)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
