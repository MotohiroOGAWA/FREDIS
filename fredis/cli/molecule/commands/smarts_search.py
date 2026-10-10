"""Search an MSDataset for a SMARTS substructure."""
import json

from ...base import CLICommand


class SmartsSearchCommand(CLICommand):
    name = "smarts-search"
    help = "Count SMARTS matches in an .msds dataset."
    description = (
        "Report matching records and unique molecules as JSON. Percentages use "
        "valid SMILES as the denominator; missing and invalid values are reported separately."
    )

    def configure(self, parser):
        parser.add_argument("--input", required=True, help="Input .msds file.")
        parser.add_argument("--smarts", required=True, action="append", help="Reactant SMARTS query. Repeat for multiple patterns; overall counts use any match.")
        parser.add_argument("--smiles-column", default="SMILES", help="SMILES metadata column (default: SMILES).")

        parser.add_argument("--include-compounds", action="store_true",
                            help="Include all valid unique compounds with canonical SMILES, match status, and record counts.")

    def run(self, args):
        from fredis.domain.molecule.smarts_search import search

        try:
            result = search(args.input, args.smarts, args.smiles_column, include_compounds=args.include_compounds)
        except Exception as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(result, indent=2, allow_nan=False))
