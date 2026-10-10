Workbench embeds the complete viewer from the `fredis/libs/msentity` submodule.
Its original JavaScript and CSS run in an iframe. The Workbench-owned `host.js`
loads the original extension module without changing its source, captures the
provider through its existing VS Code registration, and supplies a transport
facade. The original provider starts the original Python backend and handles
all viewer messages. There are no maintained FREDIS copies of chemistry or viewer behavior.

`upstream.js` uses `fredis/libs/msentity/vscode-extension` directly, including
workspace source discovery when Workbench itself is installed. It never uses an
installed `motohiroogawa.msentity-spectrum-viewer` extension or downloads a VSIX.

`npm run build:spectrum-viewer` copies the viewer host, media, Python files and
license from the submodule into generated `vendor/msentity-spectrum-viewer`.
VS Code packaging runs this via `vscode:prepublish`. This snapshot is the fallback
when the source submodule is not available. Rebuild Workbench after submodule
changes to update that packaged snapshot.

The adapter supplies `fredis.pythonPath`, the Workbench page-size setting,
a separate message channel, and disables floating spectrum windows. Chemistry
settings, structure rendering and SMARTS filtering belong to msentity.

The submodule is read-only. The host uses a temporary bootstrap dataset, removes
it immediately through the existing backend protocol, and queues startup messages
until the iframe is ready. No embedding API or patch in msentity is required.
