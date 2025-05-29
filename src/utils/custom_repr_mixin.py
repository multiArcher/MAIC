from dataclasses import fields, asdict, is_dataclass
from pprint import pformat


class CustomConfigReprMixin:
    """A mixin class that provides a custom __repr__ method for dataclasses."""

    def __repr__(self):
        lines = []
        for f in fields(self):
            # Walk through all fields and print their values.
            value = getattr(self, f.name)  # Get the value of the field.
            if is_dataclass(value):
                # If the value is a dataclass, print its fields with depth 1.
                sub_lines = []
                for sub_field_name, sub_field_value in asdict(value).items():
                    # print the sub-fields.
                    sub_lines.append(f"\t{sub_field_name} = {pformat(sub_field_value)}, ")
                sub_lines.sort()
                line = f"\t{f.name}: {value.__class__.__name__}(\n\t" + "\n\t".join(sub_lines) + "\n\t)"

            else:
                # Otherwise, print the value using pformat.
                line = f"\t{f.name} = {pformat(value, compact=True)}"

            lines.append(line)
        lines.sort()

        return f"{self.__class__.__name__}(\n" + "\n".join(lines) + "\n)"
