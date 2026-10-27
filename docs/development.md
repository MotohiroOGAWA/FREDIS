# Development and documentation checks

Documentation lives in `docs/` and uses Sphinx, Furo, MyST Markdown, Mermaid,
autodoc, implementation source links, and a local CLI reference extension. The user guides describe supported
workflows; implementation modules and argument parsers provide signatures and
option defaults.

## Build strictly

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Treat warnings as failures. Check the generated home page, navigation, API
signatures, option tables, and diagrams. The complete guide is readable from
`docs/_build/html/index.html` or through a local HTTP server.

The Python package version comes from installed distribution metadata.
CLI tables use the registered parsers through `docs/_ext/fredis_cli.py`; do not
maintain a second hand-copied complete option list. Narrative tables should
explain meaning, ownership, and important defaults.

## Execute the documented examples

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python docs/examples/verify_examples.py
```

The checker extracts runnable Bash/Python blocks from the quickstart,
fine-tuning guide, and selected API pages. It uses a fresh temporary directory
and the current Python executable, then retains commands' logs and a
`verification.json` report. `--output runs/docs-check` selects a new explicit
output directory.

The example MSDS creator preserves the same public interfaces used for measured
datasets but uses synthetic peaks. The check exercises preparation, training,
reaction export/import, fine-tuning, single/batch prediction, prepared-data
selection/persistence, and Python API calls. It is separate from the
[measured-data workflow check](guides/pattern-workflow-verification.md).

## Run regressions

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
```

Run checks appropriate to runtime changes as well as documentation examples.
GUI checks live in the separate Workbench repository. Record GPU availability
when interpreting skipped CUDA tests.

## Writing conventions

Use English for public documentation, code, comments, example strings, and
Mermaid labels. Describe concrete inputs, commands, outputs, and failure modes.
Distinguish example settings from implementation defaults and experimental
execution checks from scientific accuracy results.

Keep scientific concepts, task instructions, and complete references on their
own pages with explicit links. Add new pages to `docs/index.md` or the API
toctree so they remain reachable. Use relative links for documentation pages and
stable repository URLs for source code outside the Sphinx tree. Private
`*.local.*` notes are excluded from the public build.

Diagram sources stay in Markdown with `{mermaid}` directives. Include shape and
index-space ownership where it clarifies model/data behavior. Prefer ordinary
code blocks for copied commands and `{literalinclude}` when showing an actual
bundled example file.
