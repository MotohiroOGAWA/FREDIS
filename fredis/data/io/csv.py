from ._readers import read


def read_csv(path, **options):
    return read(path, "csv", **options)


def read_csv_text(text, **options):
    return read(text, "csv", text=True, **options)
