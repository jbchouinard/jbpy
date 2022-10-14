import math
import time
from typing import NamedTuple

import numpy as np
from PIL import Image

from jb.cli import CLI, get_input
from jb.color import Color, BandColorGradient, SmoothColorGradient


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
}
COLOR_MODES = {"band": BandColorGradient, "smooth": SmoothColorGradient}


class Point(NamedTuple):
    x: float
    y: float

    def to_complex(self) -> complex:
        return self.x + self.y * 1j

    @classmethod
    def from_complex(cls, c: complex) -> "Point":
        return cls(c.real, c.imag)


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


class Grid(NamedTuple):
    axis_x: Axis
    axis_y: Axis

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


def generate_complex_coordinates(width: int, height: int, axes: Grid) -> np.ndarray:
    pixel_box = Grid(Axis(0, width), Axis(0, height))
    arr = np.zeros((width, height), complex)

    for x in range(width):
        for y in range(height):
            p = Point(x + 0.5, y + 0.5)
            pa = pixel_box.project_to(axes, p)
            arr[x][y] = pa.to_complex()

    return arr


def i_values_colorizer(max_iterations: int, color_mode: str, palette: str):
    gradient = COLOR_MODES[color_mode](max_iterations, *PALETTES[palette])
    colors = {-1: Color(0, 0, 0)}
    for i_val in range(max_iterations + 1):
        colors[i_val] = gradient.get(i_val)

    def color_i_value(i_value, arr):
        arr[0], arr[1], arr[2] = colors[i_value]

    return color_i_value


class MandelbrotSet:
    def __init__(self, c_values: np.ndarray, treshold=2.0):
        self.iteration = 0
        self.treshold = treshold
        self.c_values = np.copy(c_values)
        self.z_values = np.copy(c_values)
        self.i_values = np.full(c_values.shape, -1)

    def iterate(self):
        self.z_values = self.z_values * self.z_values + self.c_values
        self.iteration += 1

        def update_i_values(i_value, z_value):
            if i_value != -1:
                return i_value
            assert not math.isnan(abs(z_value)), "oops, nan"
            if abs(z_value) > self.treshold:
                return self.iteration
            else:
                return -1

        self.i_values = np.vectorize(update_i_values)(self.i_values, self.z_values)

    def iterate_n(self, n: int):
        for _ in range(n):
            self.iterate()

    def to_image(self, colorizer):
        width, height = self.i_values.shape
        colors = np.zeros((height, width, 3), dtype="uint8")

        for y in range(height):
            for x in range(width):
                colorizer(self.i_values[x][y], colors[y][x])

        img = Image.fromarray(colors.astype("uint8"), mode="RGB")
        return img


cli = CLI()


@cli.command()
@cli.argument("-s", "--size", type=int, default=500)
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="colorful")
@cli.argument("-m", "--mode", choices=COLOR_MODES.keys(), default="band")
def test_color(size, color, mode):
    pixels = np.zeros((size, size, 3), np.uint8)
    colorizer = i_values_colorizer(size, mode, color)
    for y in range(size):
        for x in range(size):
            colorizer(x, pixels[y][x])

    img = Image.fromarray(pixels, mode="RGB")
    img.show()


@cli.command()
@cli.argument("y2", type=float)
@cli.argument("y1", type=float)
@cli.argument("x2", type=float)
@cli.argument("x1", type=float)
@cli.argument("-p", "--pixels", type=int, default=1000)
@cli.argument("-n", "--iterations", type=int, default=200)
@cli.argument("-m", "--mode", choices=COLOR_MODES.keys(), default="band")
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="colorful")
def viewer(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    pixels: int,
    iterations: int,
    mode: str,
    color: str,
):
    axes = Grid(Axis(x1, x2), Axis(y1, y2))
    width = pixels
    height = round(pixels * axes.yx_ratio)
    grid = generate_complex_coordinates(width, height, axes)

    mandelbrot = MandelbrotSet(grid)
    mandelbrot.iterate_n(iterations)

    colorizer = i_values_colorizer(iterations, mode, color)
    img = mandelbrot.to_image(colorizer)
    img.show()


@cli.command()
@cli.argument("-m", "--mode", choices=COLOR_MODES.keys(), default="band")
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="colorful")
def gif_iterations(mode: str, color: str):
    axes = get_input("axes", "-2.0 1.0 -1.33333 1.33333", Grid.from_str)
    width = get_input("width", "1080", int)
    height = get_input(
        "height", str(round(width * axes.axis_y.length / axes.axis_x.length)), int
    )

    iterations = get_input("iterations", "100", int)
    duration = get_input("duration", "5", int)
    output_file = get_input("output file", "mandelbrot-iter.gif", str)

    grid = generate_complex_coordinates(width, height, axes)
    colorizer = i_values_colorizer(iterations, mode, color)
    mandelbrot = MandelbrotSet(grid)
    frames = []

    for i in range(iterations):
        start = time.time()
        mandelbrot.iterate()
        frames.append(mandelbrot.to_image(colorizer))
        print(f"generated frame {i} in {time.time() - start:.2f} seconds")

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=(1000 * duration) // iterations,
        loop=0,
    )
    print(f"GIF saved to file://{output_file}")


if __name__ == "__main__":
    cli()
