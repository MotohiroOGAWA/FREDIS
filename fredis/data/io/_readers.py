from ..native import from_native


def read(source, format, *, text=False, show_progress=True,
         normalize_intensity=None, spec_id_prefix=None, source_name=None):
    import msentity
    native = (msentity.read_text(source, format) if text else
              msentity.load(source, file_type=format, spec_id_prefix=spec_id_prefix))
    with native:
        if normalize_intensity is True or (normalize_intensity is None and format in {"msp", "mgf"}):
            native.normalize()
        return from_native(native)
