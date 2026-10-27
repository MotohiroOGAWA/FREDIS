"""Render CLI option tables from FREDIS argparse definitions during Sphinx builds."""

import argparse
from functools import lru_cache
from unittest.mock import patch

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

COMMANDS = {
    "prepare": ("train", "create-fragment-tree-data"),
    "train": ("train", "fragment-tree"),
    "train-detailed": ("train", "fragment-tree-detailed"),
    "tune": ("train", "tune-fragment-tree"),
    "trainable-ids": ("train", "trainable-ids"),
    "export-pattern": ("train", "export-pattern"),
    "reaction-pattern": ("analysis", "reaction-pattern"),
    "assign-formula": ("assign-formula", "run"),
    "descriptors": ("molecule", "descriptors"),
    "smarts-search": ("molecule", "smarts-search"),
    "split": ("train", "split-datasets"),
}


@lru_cache(maxsize=1)
def root_parser():
    from fredis.cli.main import build_parser

    return build_parser()


def command_parser(name):
    if name == "predict":
        from fredis.ml.specgen.predict_spectrum import parse_args

        captured = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, *args, **kwargs):
            captured.append(parser)
            return original(
                parser,
                [
                    "--model",
                    "checkpoint.pt",
                    "--output-dir",
                    "predicted",
                    "--smiles",
                    "CCO",
                    "--ce",
                    "20",
                    "--adduct-type",
                    "[M+H]+",
                ],
            )

        with patch.object(argparse.ArgumentParser, "parse_args", capture):
            parse_args()
        parser = captured[0]
        parser.prog = "fredis-predict-spectrum"
        return parser
    parser = root_parser()
    for part in COMMANDS[name]:
        subcommands = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        parser = subcommands.choices[part]
    return parser


class FredisCLI(SphinxDirective):
    """Insert one usage synopsis and the live parser's option/default table."""

    required_arguments = 1
    has_content = False

    def run(self):
        parser = command_parser(self.arguments[0])
        output = []
        if parser.description:
            output.append(nodes.paragraph(text=parser.description))
        output.append(nodes.literal_block(text=parser.format_usage().strip()))
        table = nodes.table(classes=["longtable"])
        group = nodes.tgroup(cols=4)
        for width in [25, 15, 15, 45]:
            group += nodes.colspec(colwidth=width)
        header = nodes.thead()
        row = nodes.row()
        for label in ["Option", "Value", "Default", "Description"]:
            row += nodes.entry("", nodes.paragraph(text=label))
        header += row
        group += header
        body = nodes.tbody()
        for action in parser._actions:
            if (
                not action.option_strings
                or action.dest == "help"
                or action.help == argparse.SUPPRESS
            ):
                continue
            value = action.metavar or getattr(action.type, "__name__", "text")
            if isinstance(value, tuple):
                value = " ".join(value)
            if action.nargs == 0:
                value = "flag"
            if action.choices is not None:
                value = ", ".join(map(str, action.choices))
            default = "required" if action.required else repr(action.default)
            if action.default == argparse.SUPPRESS:
                default = "not stored"
            description = (action.help or "").replace(
                "%(default)s", str(action.default)
            )
            row = nodes.row()
            for index, text in enumerate(
                [", ".join(action.option_strings), str(value), default, description]
            ):
                entry = nodes.entry()
                paragraph = nodes.paragraph()
                paragraph += (
                    nodes.literal(text=text) if index in (0, 2) else nodes.Text(text)
                )
                entry += paragraph
                row += entry
            body += row
        group += body
        table += group
        output.append(table)
        return output


def setup(app):
    app.add_directive("fredis-cli", FredisCLI)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": True}
