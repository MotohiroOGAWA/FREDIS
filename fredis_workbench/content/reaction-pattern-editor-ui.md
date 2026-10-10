# Reaction Pattern Editor

The Workbench editor reuses the SVG atom/bond renderer and hit targets, RDKit parsing and SMARTS serialization, Webview chemistry messaging, atom maps, pattern storage, and shared selection, pan/zoom and periodic-table controls.

## Step 1

New input defaults to SMILES. Existing mapped patterns open as SMARTS. Draw Structure displays RDKit coordinates with a single scale on both axes, preserving angles and relative bond lengths. The molecule canvas uses VS Code's editor background.

Click toggles atom/bond selection; Shift-click or Shift-drag removes items. Lasso is the default; the box icon switches the drag region. Right-drag pans and the wheel zooms in either mode. Edit mode only inspects already selected items, without changing selection. Constraint changes in Edit apply to that item; selection mode permits batch constraints.

Atom matching supports Any/Custom, independent Non-H and multiple allowed elements (OR). Exclude selected elements (NOT) changes the list into an exclusion set: choosing C, N and O matches none of those elements. The serializer uses AND between negated atomic numbers (`[!#6;!#7;!#8]`), with Non-H applied independently. Periodic-table highlights retain the excluded selections, and reopening a saved exclusion SMARTS restores the list and NOT setting. An empty exclusion set permits all elements (subject to Non-H). Bond matching supports Any/Custom, multiple bond types (OR) and ring predicates. Unspecified source bond ring status follows RDKit's topology. Periodic-table buttons retain selection highlights, including mixed selections. Both pickers share 18-group data; narrow inline dialogs scroll horizontally.

Generate SMARTS generates and validates without registering a pattern or leaving Step 1. Apply validates the reactant into the local pattern draft, then collapses Step 1 and opens Step 2; it does not register anything in the Pattern Set. Opening either accordion header closes the other large editor.

## Step 2: products from a selected reactant graph

The product canvas always starts from the confirmed mapped Reactant SMARTS. All atoms and bonds are initially selected, and both default to Unchanged. There is no Product source input, input type picker, Draw Product or generic atom/bond deletion control. Canvas atom labels show map numbers; the serializer retains original atom types and queries by default.

The selected atoms and bonds define the output. Deselecting either endpoint immediately clears the selection of every incident bond, including newly added bonds. This applies to click toggling, Shift-click, Shift-lasso/box and the Selected Items controls in both steps. The other endpoint stays selected; added bonds remain in the graph. Deselecting a bond excludes the bond while keeping selected endpoints. Disconnected components are serialized in one Product SMARTS separated by `.`. An empty atom selection reports an error.

The bond + tool connects two clicked atoms with a selected single bond; reconnecting an existing deselected bond includes it again. Newly added bonds support the same click/drag selection and deselection as original bonds. In Edit mode, clicking a selected added bond exposes its type and a control to remove that new bond from the graph. Original bonds are excluded by deselection. Undo restores graph edits and inclusion selections.

Edit mode keeps selections unchanged. Product atom defaults to Unchanged; Custom permits an explicit atom SMARTS or periodic-table edit. Product bond defaults to Unchanged; Single/Double/Triple/Aromatic explicitly replaces its type. Choosing Unchanged clears the override and restores the Reactant query (new bonds have only explicit types, with Single selected initially). Atom maps remain fixed to the source.

Generate SMARTS validates and displays the selection while staying in Step 2. The + immediately adds a new unsaved Product to the Editor-local list and opens it for editing. Save validates and overwrites the selected Product; it never appends another entry. New Products and Products changed since their last Save have a * beside their names. Undoing back to the saved state clears the marker. Switching Products preserves unsaved edits without saving them. Restore Saved discards edits to the active Product and restores its saved name, atom/bond changes, inclusion and SMARTS; it is unavailable for a Product that has never been saved. Async Product loading keeps the current entry active until chemistry is ready, and stale loads or output updates cannot overwrite another Product. The compact list only shows names, unsaved markers and Remove controls; clicking an entry displays its structure, constraints, counts and outputs in the Editor. Deleting the active Product selects a remaining entry, or hides the Product editor when none remain. Generate SMARTS and Save have equal-sized buttons in one row.

The canvas count shows atoms, included bonds and the number of connected molecules immediately, including isolated selected atoms and newly added selected bonds. Product SMARTS and Reaction SMIRKS appear as two stacked read-only output fields, each with its own Copy button. There are no output tabs or Reaction SMARTS controls.

Step 3 reviews the local Pattern and provides Add Reaction Pattern to Set, which validates and commits the entire local pattern (reactant and added products) to the current Pattern Set. Editing an existing pattern clones it first; registration updates that entry, while later draft edits/deletions leave the registered entry intact until registration is pressed again. Save all new or modified Products before registering; registration never implicitly saves Product edits. Closing an unregistered draft does not change the Pattern Set. Add and Save at least one Product before registering. Continue to Step 3 moves from Product editing to registration; the three steps open one at a time. Successful registration automatically closes the Editor and leaves a confirmation beside the Pattern list; validation failures keep it open.

Copy uses the VS Code clipboard. Reaction SMIRKS currently composes the reactant and current product with `>>`; strict conversion of arbitrary SMARTS to SMIRKS remains unsupported.

## State and serialization

Source constraints remain structured objects: atom mode/elements/nonHydrogen/raw query; bond mode/types/ringStatus. A local pattern draft separates Editor changes from the registered Pattern Set. Product state uses selected atom and bond sets, explicit atom overrides, bond overrides/raw queries, added-bond records and an undo history. Inspector focus is separate from selection.

`productFromSelection` preserves source queries, applies explicit edits, removes excluded edges and serializes selected mapped atoms with RDKit. The shared fragment helper explicitly removes excluded bonds because RDKit interprets an empty `bondsToUse` list as all bonds. `productState` recovers selected atoms/bonds, changed atom/bond queries and added bonds when a saved product is reopened.

## Changed files and validation

- `src/features/reaction-pattern-set/visual-editor.js`: Workbench components, state, interaction and product-selection UI.
- `src/features/reaction-pattern-set/backend.py`: structured constraints, selection-based product serialization and recovery.
- `src/features/reaction-pattern-set/periodic-table.js`: shared 18-group element layout.
- `src/extension.js`: UI installation, shared standalone periodic table, source defaults and clipboard messaging.
- `scripts/check-reaction-pattern-ui.js`: jsdom interaction and real-RDKit integration checks.
- `scripts/check-reaction-pattern-visual-products.py`: query semantics, product serialization and state recovery checks.
- `package.json` / `package-lock.json`: UI check integration and synchronization of the declared jsdom dependency.

Passed: reaction-pattern UI tests, RDKit chemistry tests and `npm run check:workbench`. Tests cover unchanged atom/bond queries, dot-separated fragments (including no selected bonds), selected subsets and maps, explicit atom/bond edits, selecting/deselecting/removing added bonds, Undo, saved-product recovery, periodic-table highlights/positions, Edit selection isolation, SMILES reset defaults, and both generation buttons staying in their respective steps, local product add/remove/preview, stacked output copying, molecule counts and explicit Pattern Set registration/update isolation. Integration also uses the exact `CC=CC1CCCCC1` source with real RDKit.

The earlier full `npm run check` stopped at a missing `score-distribution` module; individually run fine-tune and step-validation checks also exposed unrelated assertion failures. No browser/VS Code screenshot verification was performed. Atom creation and charge-specific tools remain future work.

All Workbench buttons provide hover, pressed/click and keyboard-focus feedback. Async Editor actions show a busy spinner; mutation buttons prevent duplicate submissions while processing. Motion effects respect reduced-motion preferences.

## Direct reaction preview

Reaction Preview beside Save Configuration tests the current Pattern Set. Each Pattern card has its own Reaction Preview entry point; the preview target can also be switched between the Set and one Pattern. Enter SMILES and run the preview. Every mode shows Pattern filters on the left, the input molecule drawn by RDKit, `Matched Reactants`, and products. There is no separate `Matched Reactions` list. Selecting a reactant opens its Product definitions and concrete molecule sets. Clicking a highlighted bond filters `Matched Reactants` to matches containing that bond; clicking it again clears the filter. Overlapping sites provide explicit choices, and symmetric query-role assignments at the same site are grouped while retaining distinct generated results.

Each concrete molecule has its own RDKit SVG, formula, exact mass and copyable Product SMILES. When one Product SMARTS produces dot-separated molecules, they appear as `Product name (1)`, `Product name (2)`, and so on, with separate Product SMILES; Product SMARTS and the Source-specific Reaction SMIRKS remain shared once for the group. Product-definition serial numbers precede the name. Stepwise candidate and current-fragment cards expose the same Source-specific Reaction SMIRKS format used by exhaustive candidates. Stepwise has no Max reactions control. Exhaustive accepts `-1` to use the primitive-reaction count as its effective upper bound. The initial direct-preview request only parses Reactant SMARTS, matches the input and draws the input molecule. Reactions are compiled and executed only when a site is selected. Each query role is bound to its original input index before running the actual SMIRKS, so omitted roles and symmetric assignments cannot cause products from another site to be displayed. Source atom indices are preserved even when canonical SMILES ordering differs. SVG foreground and canvas background follow the Workbench theme.

Malformed Patterns and unsanitizable products report errors without replacing them with template drawings or blocking valid Patterns in the Set. No-match, empty selection and invalid SMILES states are explicit. Pattern or input edits invalidate the preview, and stale requests cannot replace newer results. Enumeration is limited to 512 query assignments and 512 generated groups per selected binding/rule, with a visible limit notice. A persistent Python/RDKit worker is reused for matching and selected-site generation; it is stopped when its owning Workbench closes and restarted after unexpected exit. Parsed queries, compiled SMIRKS, identical molecule depictions and selected-site results are cached. Previewing never edits the Pattern Set or its configuration. Run `npm run check:reaction-preview` for actual RDKit/backend and UI checks.
