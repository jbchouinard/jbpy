from typing import NamedTuple


def interpolate(axis_min, axis_max, scale, value):
    return axis_min + (axis_max - axis_min) * (value / scale)


class Color(NamedTuple):
    red: int
    green: int
    blue: int

    def mix(self: "Color", other: "Color", strength: float = 0.5) -> "Color":
        return Color(
            int(interpolate(self.red, other.red, 1.0, strength)),
            int(interpolate(self.green, other.green, 1.0, strength)),
            int(interpolate(self.blue, other.blue, 1.0, strength)),
        )

    @classmethod
    def from_hex(cls, s: str) -> "Color":
        assert s.startswith("#")
        return Color(int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))


class BandColorGradient:
    def __init__(self, scale: float, *colors: Color):
        self.step = scale / len(colors)
        self.colors = list(colors)

    def get(self, value):
        idx = min(len(self.colors) - 1, round(value / self.step))
        return self.colors[idx]


class SmoothColorGradient:
    def __init__(self, scale: float, *colors: Color):
        self.colors = list(zip(colors[:-1], colors[1:]))
        assert self.colors
        self.step = int(scale) // len(self.colors)

    def get(self, value):
        idx = min(len(self.colors) - 1, int(value / self.step))
        c1, c2 = self.colors[idx]
        s = (value - (idx * self.step)) / self.step
        return c1.mix(c2, max(0.0, min(1.0, s)))


WHITE = Color.from_hex("#ffffff")
BLACK = Color.from_hex("#000000")
PALETTES = {
    "bw": [BLACK, WHITE],
    "colorful": [
        Color.from_hex("#003f5c"),
        Color.from_hex("#2f4b7c"),
        Color.from_hex("#665191"),
        Color.from_hex("#a05195"),
        Color.from_hex("#d45087"),
        Color.from_hex("#f95d6a"),
        Color.from_hex("#ff7c43"),
        Color.from_hex("#ffa600"),
    ],
    "rainbow": [
        Color.from_hex("#ff0000"),
        Color.from_hex("#ff8700"),
        Color.from_hex("#ffd300"),
        Color.from_hex("#deff0a"),
        Color.from_hex("#a1ff0a"),
        Color.from_hex("#0aff99"),
        Color.from_hex("#0aefff"),
        Color.from_hex("#147df5"),
        Color.from_hex("#580aff"),
        Color.from_hex("#be0aff"),
    ],
}
GRADIENT_MODES = {"band": BandColorGradient, "smooth": SmoothColorGradient}


def make_gradient(scale: float, palette: str, mode: str):
    cls = GRADIENT_MODES[mode]
    colors = PALETTES[palette]
    return cls(scale, *colors)
