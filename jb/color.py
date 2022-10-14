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
