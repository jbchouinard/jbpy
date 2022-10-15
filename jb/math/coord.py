from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float

    def to_complex(self) -> complex:
        return self.x + self.y * 1j

    @classmethod
    def from_complex(cls, c: complex) -> "Point":
        return cls(c.real, c.imag)

    def interpolate_linear(self, other: "Point", frac: float) -> "Point":
        interpolate = linear_interpolator(frac)
        return Point(interpolate(self.x, other.x), interpolate(self.y, other.y))


class Axis(NamedTuple):
    minv: float
    maxv: float

    @property
    def length(self) -> float:
        return self.maxv - self.minv

    def bound(self, v):
        return max(self.minv, min(self.maxv, v))

    def project_to(self, other: "Axis", v: float) -> float:
        frac = (self.bound(v) - self.minv) / self.length
        return other.minv + frac * other.length

    def __str__(self):
        return f"{self.minv}-{self.maxv}"


def linear_interpolator(frac):
    def interpolate(x, y):
        return x * (1.0 - frac) + y * frac

    return interpolate


def geometric_interpolator(frac):
    def interpolate(x, y):
        return x ** (1.0 - frac) * y ** (frac)

    return interpolate


class Grid(NamedTuple):
    axis_x: Axis
    axis_y: Axis

    @classmethod
    def from_floats(cls, x1, x2, y1, y2) -> "Grid":
        return cls(Axis(x1, x2), Axis(y1, y2))

    def bound(self, p: Point) -> "Point":
        return Point(x=self.axis_x.bound(p.x), y=self.axis_y.bound(p.y))

    def project_to(self, other: "Grid", p: Point) -> Point:
        x = self.axis_x.project_to(other.axis_x, p.x)
        y = self.axis_y.project_to(other.axis_y, p.y)
        return Point(x, y)

    @property
    def yx_ratio(self) -> float:
        return self.axis_y.length / self.axis_x.length

    def __str__(self):
        return f"{self.axis_x},{self.axis_y}"

    @classmethod
    def from_str(cls, s: str) -> "Grid":
        parts = s.split(",")
        assert len(parts) == 4, "expected 4 values"
        parts = [float(x) for x in parts]
        return cls(Axis(*parts[:2]), Axis(*parts[2:]))

    def to_window(self) -> "Window":
        h = self.axis_y.maxv - self.axis_y.minv
        w = self.axis_x.maxv - self.axis_x.minv
        x = (self.axis_x.minv + self.axis_x.maxv) / 2.0
        y = (self.axis_y.minv + self.axis_y.maxv) / 2.0
        return Window(Point(x, y), h, w)

    def interpolate_linear(self, other: "Grid", frac: float) -> "Grid":
        interpolate = linear_interpolator(frac)
        return Grid.from_floats(
            interpolate(self.axis_x.minv, other.axis_x.minv),
            interpolate(self.axis_x.maxv, other.axis_x.maxv),
            interpolate(self.axis_y.minv, other.axis_y.minv),
            interpolate(self.axis_y.maxv, other.axis_y.maxv),
        )

    def interpolate_geometric(self, other: "Grid", frac: float) -> "Grid":
        return self.to_window().interpolate_geometric(other.to_window(), frac).to_grid()


class Window(NamedTuple):
    position: Point
    height: float
    width: float

    def to_grid(self) -> Grid:
        min_x = self.position.x - self.width / 2.0
        max_x = self.position.x + self.width / 2.0
        min_y = self.position.y - self.height / 2.0
        max_y = self.position.y + self.height / 2.0
        return Grid(Axis(min_x, max_x), Axis(min_y, max_y))

    def interpolate_geometric(self, other: "Window", frac: float) -> "Window":
        interpolate = geometric_interpolator(frac)
        h = interpolate(self.height, other.height)
        w = interpolate(self.width, other.width)
        linear_frac = 1.0 - ((h - other.height) / (self.height - other.height))
        p = self.position.interpolate_linear(other.position, linear_frac)
        return Window(p, h, w)
