# Text formats: MSP, MGF, TSV, CSV

How msentity reads and writes the text interchange formats. Every example on
this page is real input or output of the library:

* [`docs/examples/plain.msp`](../examples/plain.msp) — two ordinary spectra (only m/z and intensity),
* [`docs/examples/example.msp`](../examples/example.msp) — the same spectra with a fragment
  formula on some peaks (§5).

Text is UTF-8. Lines may end with `\n`, `\r\n`, or `\r`.

---

## 1. Overview

| Format | One spectrum is | Metadata | Peaks |
|---|---|---|---|
| MSP | a block ending with a blank line | `Key: value` lines | one `m/z intensity` line per peak after `Num Peaks:` |
| MGF | `BEGIN IONS` … `END IONS` | `KEY=value` lines | one `m/z intensity` line per peak after the metadata |
| TSV | one row (tab-separated) | one column per key | `Peak` column: `mz,intensity;mz,intensity;…` |
| CSV | one row (comma-separated) | one column per key | `Peak` column: `mz,intensity;mz,intensity;…` |

All four formats can additionally carry **per-peak metadata** such as a
fragment formula; this is optional and described in §5.

Reading any of them produces a dataset with:

| Part | Content |
|---|---|
| spectrum table | one column per metadata key, in order of first appearance |
| peaks | m/z and intensity of every peak |
| peak table | only if the file has per-peak metadata (§5) |

---

## 2. MSP

### 2.1 Reading

Input (`plain.msp`, first record):

```text
NAME: MassSpecID0000001          ← metadata line "key: value"
PRECURSORMZ: 288.1225
PRECURSORTYPE: [M+H]1+
FORMULA: C16H17NO4
COLLISIONENERGY: 30.0
Num Peaks: 4                     ← starts the peaks
91.0542	245                      ← m/z <whitespace> intensity
125.0233	1000
185.0961	349
246.1125	735
                                 ← blank line: end of this spectrum
```

Result (both records):

| Name | PrecursorMZ | AdductType | Formula | CollisionEnergy | NumPeaks |
|---|---|---|---|---|---|
| MassSpecID0000001 | 288.1225 | [M+H]+ | C16H17NO4 | 30.0 | 4 |
| MassSpecID0000002 | 288.1225 | [M+H]+ | C16H17NO4 | 20.0 | 3 |

| spectrum | m/z | intensity |
|---|---|---|
| 0 | 91.0542, 125.0233, 185.0961, 246.1125 | 0.245, 1.0, 0.349, 0.735 |
| 1 | 91.0542, 125.0233, 246.1125 | 0.099, 0.281, 1.0 |

What happened:

| Input | Result | Rule |
|---|---|---|
| `NAME`, `FORMULA`, … | `Name`, `Formula`, … | header names are canonicalized (§6) |
| `PRECURSORTYPE` | `AdductType` | alias (§6) |
| `[M+H]1+` | `[M+H]+` | adduct values are normalized (§6) |
| `Num Peaks` | `NumPeaks` | canonicalized; its value is kept as a column |
| intensity `245` | `0.245` | each spectrum is divided by its highest peak (1000); can be switched off |
| `288.1225`, `30.0` | text `"288.1225"`, `"30.0"` | MSP values are kept as text |
| key missing in a record | `""` | |

Rules:

| Situation | Behavior |
|---|---|
| `key: value` | split at the **first** `:`; key and value are stripped |
| line without `:` before the peaks | error → the record is discarded (§7) |
| canonical key `NumPeaks` | the following lines are peaks |
| blank lines before `NumPeaks` | ignored |
| blank line after the peaks started | end of the record |
| end of file | ends the current record (no final blank line needed) |
| the same key twice in a record (after canonicalization) | error → record discarded; with *allow duplicates*: renamed `Name1`, `Name2`, … |
| peak line | m/z and intensity separated by whitespace (spaces or tabs) |

### 2.2 Writing

Output of `write_msp` for the dataset above (first record):

```text
Name: MassSpecID0000001          ← every visible column except NumPeaks
PrecursorMZ: 288.1225
AdductType: [M+H]+
Formula: C16H17NO4
CollisionEnergy: 30.0
NumPeaks: 4                      ← always the actual number of peaks
91.0542	0.245                    ← m/z TAB intensity
125.0233	1.0
185.0961	0.349
246.1125	0.735
                                 ← blank line after each record
```

| Item | Format |
|---|---|
| header name | the column name, or the name given in the writer's header map |
| metadata value | value text (§8) |
| m/z, intensity | float text (§8): `0.245`, `1.0`, `1e-05` |
| separator between m/z and intensity | TAB (configurable) |

---

## 3. MGF

### 3.1 Reading

```text
BEGIN IONS                       ← start of a spectrum (any letter case)
TITLE=MassSpecID0000001          ← metadata "key=value", split at the first "="
PEPMASS=288.1225
91.0542 245                      ← first line without "=": peaks start here
125.0233 1000
END IONS                         ← end of the spectrum
```

| Situation | Behavior |
|---|---|
| lines outside `BEGIN IONS` … `END IONS` | ignored |
| line with `=` before the first peak | metadata |
| first line without `=` | first peak; every later line of the record is a peak |
| `BEGIN IONS` while a record is open, or end of file | closes the open record |
| record with peaks but no metadata | a spectrum without metadata values |

Header canonicalization (§6) and intensity scaling apply as for MSP.

### 3.2 Writing

```text
BEGIN IONS
Name=MassSpecID0000001
PrecursorMZ=288.1225
AdductType=[M+H]+
Formula=C16H17NO4
CollisionEnergy=30.0
91.0542	0.245
125.0233	1.0
185.0961	0.349
246.1125	0.735
END IONS
                                 ← blank line
```

As MSP, but `key=value` and no `NumPeaks` line (an existing `NumPeaks`
column is not written).

---

## 4. TSV and CSV

### 4.1 Layout

```text
Name	PrecursorMZ	AdductType	Formula	CollisionEnergy	NumPeaks	Peak
MassSpecID0000001	288.1225	[M+H]+	C16H17NO4	30.0	4	91.054199999999994,0.245;125.02330000000001,1;…
MassSpecID0000002	288.1225	[M+H]+	C16H17NO4	20.0	3	91.054199999999994,0.099000000000000005;…
```

* One spectrum per row. The first row holds the column names; they must be
  unique and include `Peak`.
* `Peak` lists the peaks as `mz,intensity` pairs separated by `;`
  (empty cell = no peaks). Spaces around numbers are allowed:
  `91.0542, 245; 125.0233, 1000` is valid.
* Every other column is spectrum metadata.

The CSV version is identical with `,` as delimiter; the `Peak` cell is then
quoted because it contains commas:

```text
Name,PrecursorMZ,AdductType,Formula,CollisionEnergy,NumPeaks,Peak
MassSpecID0000001,288.1225,[M+H]+,C16H17NO4,30.0,4,"91.054199999999994,0.245;125.02330000000001,1;…"
```

### 4.2 Reading — column types

Example file:

```text
Name	PrecursorMZ	Charge	ID	Peak
A	288.1225	1	007	91.0542,0.245;125.0233,1
B		2	008
```

| Column | default: text as written | with type inference |
|---|---|---|
| Name | `"A"`, `"B"` | `"A"`, `"B"` (text) |
| PrecursorMZ | `"288.1225"`, `""` | `288.1225`, *missing* (float) |
| Charge | `"1"`, `"2"` | `1`, `2` (integer) |
| ID | `"007"`, `"008"` | `7`, `8` (integer — leading zeros lost) |

Type inference (option `infer_types`, CLI `--infer-types`) decides per column:

| Non-empty cells | Empty cells | Column type |
|---|---|---|
| all integers | none | integer |
| all numbers (§8, `nan` excluded) | any | float; empty cells missing |
| anything else | any | text; empty cells missing |
| (none) | all | float, all missing |

### 4.3 Reading — quoting and rows

| Input field | Value |
|---|---|
| `"a, b"` | `a, b` (delimiter inside quotes) |
| `"say ""hi"""` | `say "hi"` (doubled quote) |
| `"line 1⏎line 2"` | a value containing a line break |
| `ab"c` | `ab"c` (a quote inside an unquoted field is ordinary text) |

| Situation | Behavior |
|---|---|
| UTF-8 byte-order mark | ignored |
| empty line | skipped |
| row with fewer fields | missing fields are empty |
| row with more fields than the header | error |
| malformed `Peak` (`1;2`, `1,a`) | error |

### 4.4 Writing

| Item | Format |
|---|---|
| metadata cell | value text (§8); a missing cell of a numeric column is written `nan` |
| peaks | `printf("%.17g")` for both numbers, so every double reads back exactly (`91.0542` → `91.054199999999994`) |
| quoting | only fields containing the delimiter, `"`, CR, or LF; inner `"` doubled |
| line ending | `\n` |

---

## 5. Per-peak metadata (optional)

Each peak can carry extra values, for example the formula of a fragment.
They form the **peak table** of the dataset (one text column per name).

### 5.1 MSP and MGF: the peak column line

The first line of the peaks may name extra columns. It starts with the
words `mz intensity` (any letter case), followed by the names. Every peak line
can then have one more item per name, separated from the intensity by
whitespace:

```text
Num Peaks: 4
mz intensity formula             ← peak column line: one extra column "formula"
91.0542	245	C7H7+                ← formula = C7H7+
125.0233	1000	C6H5O3+          ← formula = C6H5O3+
185.0961	349                      ← no value → ""
246.1125	735
```

Result (`example.msp`):

| peak | m/z | intensity | formula |
|---|---|---|---|
| 0 | 91.0542 | 0.245 | C7H7+ |
| 1 | 125.0233 | 1.0 | C6H5O3+ |
| 2 | 185.0961 | 0.349 | |
| 3 | 246.1125 | 0.735 | |
| 4 | 91.0542 | 0.099 | C7H7+ |
| 5 | 125.0233 | 0.281 | |
| 6 | 246.1125 | 1.0 | |

Several columns are separated by `;`; a value may be quoted with `"` or `'`
to contain `;`:

```text
mz intensity formula note
125.0233  1000  C6H5O3+ ; "loss; H2O"      ← formula = C6H5O3+, note = loss; H2O
```

| Step | Result |
|---|---|
| split the line at the first two runs of whitespace | `125.0233`, `1000`, `C6H5O3+ ; "loss; H2O"` |
| split the rest at `;`, re-join quoted parts, remove the quotes | `C6H5O3+`, `loss; H2O` |
| assign the names of the peak column line | `formula = C6H5O3+`, `note = loss; H2O` |
| items beyond the named columns, or without a peak column line | named `column1`, `column2`, … |

| Situation | Behavior |
|---|---|
| no peak column line | peaks are just m/z and intensity; extra items become `column1`, … |
| the peak column line | applies to the current spectrum only |
| empty item | that peak has `""` in that column |
| the same column twice in one line (including `mz`, `intensity`) | error → record discarded |

Writing: when at least one peak of a spectrum has a value in a selected peak
column, the writer adds the peak column line with the columns that have
values in that spectrum, and quotes the values:

```text
NumPeaks: 4
mz	intensity	formula
91.0542	0.245	"C7H7+"
125.0233	1.0	"C6H5O3+"
185.0961	0.349                    ← all values empty → nothing after intensity
246.1125	0.735
```

Several columns are written as `"value1" ; "value2"`; a `"` inside a value
becomes `'`. Spectra without values are written exactly as in §2.2.

### 5.2 TSV and CSV: column names at the start of the Peak cell

Without per-peak metadata the `Peak` cell is `mz,intensity;mz,intensity;…`
(§4). With per-peak metadata, the cell **starts with an entry naming the
fields** — the same idea as the MSP peak column line — and every peak entry
has the extra values in that order:

```text
Peak cell = mz,intensity,formula;91.0542,245,C7H7+;125.0233,1000,C6H5O3+;185.0961,349
            └──── field names ──┘└──── peak 0 ────┘└────── peak 1 ──────┘└─ peak 2 ─┘
```

Output of `write_tsv` for `example.msp`:

```text
Name	PrecursorMZ	AdductType	Formula	CollisionEnergy	NumPeaks	Peak
MassSpecID0000001	…	4	mz,intensity,formula;91.054199999999994,0.245,C7H7+;125.02330000000001,1,C6H5O3+;185.09610000000001,0.34899999999999998;246.11250000000001,0.73499999999999999
MassSpecID0000002	…	3	mz,intensity,formula;91.054199999999994,0.099000000000000005,C7H7+;125.02330000000001,0.28100000000000003;246.11250000000001,1
```

| Rule | Example |
|---|---|
| the first entry names the fields if its first two fields are `mz` and `intensity` (any case) | `mz,intensity,formula,note` |
| a peak entry has 2 up to 2 + (number of names) fields; missing trailing values are empty | `185.0961,349` → formula `""` |
| an empty value in the middle is an empty field | `125.0233,1000,,Y` → formula `""`, note `Y` |
| a value containing `,` `;` `"` or leading/trailing spaces is quoted, `"` doubled | `1,2,"loss; H2O, ""x"""` → `loss; H2O, "x"` |
| a cell without the name entry has no per-peak metadata | `91.0542,245;125.0233,1000` |
| rows may name different fields; the peak table has the union, `""` where a row has no value | |
| writing: the name entry lists only the columns with a value in that spectrum; a spectrum without values is written plainly | |

Errors: more fields than names, a repeated or reserved name (`mz`,
`intensity`), an unclosed quote, or text after a closing quote.

The `Peak` cell is quoted once more by the TSV/CSV rules when it contains the
delimiter or `"` (always in CSV, because it contains commas).

---

## 6. Header and adduct canonicalization (MSP, MGF)

Enabled by default; each can be switched off when reading.

### 6.1 Header names

| Step | `precursor_type` | `NAME` | `Num Peaks` | `DB#` |
|---|---|---|---|---|
| 1. split at `_`/whitespace and before every capital letter | `precursor`, `type` | `N`,`A`,`M`,`E` | `Num`, `Peaks` | `D`, `B#` |
| 2. capitalize each part and join ("capitalized") | `PrecursorType` | `NAME` | `NumPeaks` | `DB#` |
| 3. remove `/`, lower-case ("normalized") | `precursortype` | `name` | `numpeaks` | `db#` |
| 4. look up the alias table | **`AdductType`** | **`Name`** | **`NumPeaks`** | — |
| result | `AdductType` | `Name` | `NumPeaks` | `DB#` (step 2 spelling) |

Alias table (looked up by normalized form, so `PRECURSORMZ`,
`precursor_mz`, and `PrecursorM/Z` all become `PrecursorMZ`):

| Canonical name | Also accepted |
|---|---|
| `Name`, `Formula`, `InChIKey`, `InChI`, `SpectrumType`, `InstrumentType`, `Instrument`, `IonMode`, `CollisionEnergy`, `ExactMass`, `SMILES`, `Ontology`, `NumPeaks` | any spelling with the same normalized form |
| `PrecursorMZ` | `PrecursorM/Z` |
| `AdductType` | `PrecursorType` |
| `RetentionTime` | `RT` |

A key not in the table keeps the step-2 spelling of the **first** key with
that normalized form in the same file: if `COMMENT` appears first, a later
`Comment` is also stored as `COMMENT`.

### 6.2 Adduct values (`AdductType`)

| Input | Output | Why |
|---|---|---|
| `[M+H]+` | `[M+H]+` | already canonical |
| `[M+H]1+` | `[M+H]+` | charge 1 is written as a bare sign |
| `[M+H]++` | `[M+H]2+` | two `+` signs = charge 2 |
| `[M-2H]--` | `[M-2H]2-` | two `-` signs = charge −2 |
| `[M]0+` | `[M]` | charge 0 is omitted |
| `[M + Na] +` | `[M+Na]+` | spaces removed |
| `[M+H]x` | `[M+H]x` | not a charge: text after `]` kept |
| `M+H` | `M+H` | no brackets: only spaces removed |

Rule: the charge is the text after the first `]` that is followed, to the end,
by optional digits and one or more `+`/`-` signs. Only `+` signs → that many
positive; only `-` → that many negative; digits then a sign → ± the digits;
otherwise ±1 by the last sign. Written as nothing (0), `+`/`-` (±1), or
`n+`/`n-`.

---

## 7. Errors in MSP/MGF records

Any error inside a record discards the **whole record**; reading continues
with the next one. Optionally the errors are logged.

Input:

```text
Name: bad
NAME: again          ← NAME and Name are the same key → error
NumPeaks: 1
1 1

Name: good           ← next record is read normally
NumPeaks: 1
2 2
```

Result: one spectrum (`good`) and this log entry:

```text
[ERROR] Line (00002) NAME: again | Duplicate meta key: (NAME & bad) -> Name
[CONTEXT]                                    ← only at the "detail" log level
Name: bad                                    ← the record's lines (last 10)
NAME: again
NumPeaks: 1
1 1
                                             ← blank line between entries
```

| Setting | Default |
|---|---|
| log level | none (basic = `[ERROR]` lines, detail = with `[CONTEXT]`) |
| log file | `<input without extension>_error_<YYYYmmddHHMMSS>.txt`, `_1`, `_2`, … if it exists |

---

## 8. Text and number conventions

| Term | Definition | Examples |
|---|---|---|
| whitespace | space, TAB, LF, VT, FF, CR, and U+001C–U+001F | |
| strip | remove whitespace at both ends | `"  a b "` → `"a b"` |
| number syntax (reading) | optional sign; digits with optional fraction and exponent, a single `_` allowed between digits; or `inf`/`infinity`/`nan` in any case; surrounding whitespace allowed | `12`, `-1.5`, `.5`, `5.`, `1e-3`, `1_000`, `Infinity` |
| float text (writing) | shortest digits that read back to the same double; positional if 1e-4 ≤ \|x\| < 1e16 (at least one fractional digit), else `d.ddde±XX` | `100.0`, `0.1`, `0.0001`, `1e-05`, `1.5e+16`, `nan`, `-inf` |
| value text (writing metadata) | missing → empty; boolean → `True`/`False`; integer → digits; float → float text (NaN → empty); text → as is | `""`, `True`, `42`, `288.1225`, `C7H7+` |
