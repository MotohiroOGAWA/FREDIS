from ._readers import read


def read_tsv(path, **options):
    return read(path, "tsv", **options)


def read_tsv_text(text, **options):
    return read(text, "tsv", text=True, **options)
