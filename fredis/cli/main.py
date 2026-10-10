from __future__ import annotations

import argparse

from .discovery import register_groups_from_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fredis",
        description="Command line interface for FREDIS.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    register_groups_from_package(
        subparsers,
        package_name="fredis.cli",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
