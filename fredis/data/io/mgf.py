from ._readers import read


def read_mgf(path, **options):
    return read(path, "mgf", **options)


def read_mgf_text(text, **options):
    return read(text, "mgf", text=True, **options)
