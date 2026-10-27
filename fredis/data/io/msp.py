from ._readers import read


def read_msp(path, **options):
    return read(path, "msp", **options)


def read_msp_text(text, **options):
    return read(text, "msp", text=True, **options)
