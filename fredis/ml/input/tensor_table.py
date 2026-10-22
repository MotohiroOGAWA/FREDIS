"""Device transfer for immutable tables with tensor and optional metadata fields."""

from dataclasses import fields, replace


class TensorTable:
    def to(self, device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in fields(self)
                if hasattr(getattr(self, f.name), "to")
            },
        )
