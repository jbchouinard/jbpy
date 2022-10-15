import pathlib
import pickle
import time
from typing import List
import webbrowser

import numpy as np
from PIL import Image

from jb.cli import CLI
from jb.color import Color, GRADIENT_MODES, PALETTES, make_gradient
from jb.math.coord import Axis, Grid, Point, Window


INT16_MAX = np.iinfo(np.int16).max


def generate_complex_coordinates(width: int, height: int, axes: Grid) -> np.ndarray:
    x_start = axes.axis_x.minv
    x_scale = axes.axis_x.length / width
    x_axis = [x_start + (x + 0.5) * x_scale for x in range(width)]
    x_pixels = np.array([x_axis])

    y_start = axes.axis_y.minv
    y_scale = axes.axis_y.length / height
    y_axis = [y_start + (y + 0.5) * y_scale for y in range(height)]
    y_pixels = np.array([y_axis])

    c_grid = 1j * y_pixels + x_pixels.T
    return c_grid


def i_values_colorizer(max_iterations: int, color_mode: str, palette: str):
    gradient = make_gradient(max_iterations, palette, color_mode)

    colors = {INT16_MAX: Color(0, 0, 0)}
    for i_val in range(max_iterations + 1):
        colors[i_val] = gradient.get(i_val)

    get_color_vec = np.vectorize(colors.__getitem__)

    def color_i_values(i_values):
        return np.dstack(get_color_vec(i_values)).astype("uint8")

    return color_i_values


class MandelbrotSet:
    def __init__(self, c_values: np.ndarray, treshold=2.0):
        self.iteration = 0
        self.treshold = treshold
        self.c_values = np.copy(c_values)
        self.z_values = np.copy(c_values)
        self.i_values = np.full(c_values.shape, INT16_MAX, dtype="int16")

    def _iterate(self):
        np.square(self.z_values, out=self.z_values)
        np.add(self.z_values, self.c_values, out=self.z_values)
        self.iteration += 1

    def _update_i_values(self):
        magnitudes = np.absolute(self.z_values)
        blown_up = np.greater(magnitudes, self.treshold)
        new_i_values = blown_up * self.iteration + np.logical_not(blown_up) * INT16_MAX
        self.i_values = np.minimum(self.i_values, new_i_values)

    def iterate(self, n: int = 1, update_i_values_period: int = 1):
        for _ in range(n):
            self._iterate()
            if (self.iteration % update_i_values_period) == 0:
                self._update_i_values()

    def copy_i_values(self):
        return np.copy(self.i_values)

    @staticmethod
    def i_values_to_image(i_values, colorizer):
        return Image.fromarray(colorizer(i_values.T), mode="RGB")

    def to_image(self, colorizer):
        return self.i_values_to_image(self.i_values, colorizer)


cli = CLI()


@cli.command()
@cli.argument("-s", "--size", type=int, default=500)
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="colorful")
@cli.argument("-m", "--mode", choices=GRADIENT_MODES.keys(), default="band")
def test_color(size, color, mode):
    i_values = np.zeros((size, size), dtype=np.int16)
    with np.nditer(i_values, flags=["multi_index"], op_flags=["readwrite"]) as it:
        for i_val in it:
            x, _ = it.multi_index
            i_val[...] = x

    colorizer = i_values_colorizer(size, mode, color)
    pixels = colorizer(i_values.T)
    img = Image.fromarray(pixels, mode="RGB")
    img.show()


def generate_viewer_image(width, height, axes: Grid, iterations, colorizer):
    grid = generate_complex_coordinates(width, height, axes)
    mandelbrot = MandelbrotSet(grid)
    mandelbrot.iterate(iterations, 5)
    return mandelbrot.to_image(colorizer)


@cli.command()
@cli.argument("location", type=float, nargs=4)
@cli.argument("-p", "--pixels", type=int, default=1000)
@cli.argument("-n", "--iterations", type=int, default=200)
@cli.argument("-m", "--mode", choices=GRADIENT_MODES.keys(), default="smooth")
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="rainbow")
@cli.argument("-o", "--output-png", type=str, default="viewer.png")
def viewer(
    location: List[float],
    pixels: int,
    iterations: int,
    mode: str,
    color: str,
    output_png: str,
):
    axes = Grid.from_floats(*location)
    width = pixels
    height = round(pixels * axes.yx_ratio)
    colorizer = i_values_colorizer(iterations, mode, color)

    img = generate_viewer_image(width, height, axes, iterations, colorizer)
    img.save(output_png)
    webbrowser.open(output_png)


def loop_frames(frames, freeze=10):
    return [frames[0]] * freeze + frames[1:-1] + [frames[-1]] * freeze + frames[-1:1:-1]


@cli.command()
@cli.argument("-p", "--pixels", type=int, default=1000)
@cli.argument("-n", "--iterations", type=int, default=200)
@cli.argument("-m", "--mode", choices=GRADIENT_MODES.keys(), default="smooth")
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="rainbow")
def explorer(pixels, iterations, mode, color):
    import readline
    import webbrowser

    selection_grid = Grid.from_floats(0, 100, 0, 100)
    colorizer = i_values_colorizer(iterations, mode, color)

    location = Grid.from_floats(-2.0, 1.0, -1.5, 1.5)
    history = [location]

    def saveloc(name):
        with open(f"location_{name}.pickle", "wb") as f:
            pickle.dump(location, f)

    def loadloc(name):
        nonlocal location
        with open(f"location_{name}.pickle", "rb") as f:
            loaded = pickle.load(f)
        assert type(loaded) is Grid
        location = loaded
        redraw()

    def redraw():
        img = generate_viewer_image(pixels, pixels, location, iterations, colorizer)
        img.save("explorer.png")
        webbrowser.open(f"file://{pathlib.Path('explorer.png').resolve()}")

    def move(x, y):
        x = int(x)
        y = int(y)
        new_center = selection_grid.project_to(location, Point(x, y))
        old_window = location.to_window()
        new_window = Window(new_center, old_window.height, old_window.width)
        return new_window.to_grid()

    def zoom(factor):
        factor = float(factor)
        window = location.to_window()
        zoomed = Window(
            position=window.position,
            height=window.height / factor,
            width=window.width / factor,
        )
        return zoomed.to_grid()

    def update(newloc):
        nonlocal location
        history.append(location)
        location = newloc
        redraw()

    def make_gif(duration, fps, output_file):
        duration = int(duration)
        fps = int(fps)
        output_file = str(output_file)
        gif_zoom(
            output_gif=output_file,
            duration=duration,
            fps=fps,
            start=[-2.0, 1.0, -1.5, 1.5],
            end=[
                location.axis_x.minv,
                location.axis_x.maxv,
                location.axis_y.minv,
                location.axis_y.maxv,
            ],
            pixels=pixels,
            iterations=iterations,
            color=color,
            mode=mode,
        )

    def command():
        nonlocal location
        nonlocal history

        command, *args = input("m> ").strip().split(" ")

        if command == "undo":
            if history:
                location = history.pop()
                redraw()
        elif command == "move":
            update(move(*args))
            redraw()
        elif command == "zoom":
            update(zoom(*args))
            redraw()
        elif command == "loc":
            print(location)
        elif command == "saveloc":
            saveloc(*args)
        elif command == "loadloc":
            loadloc(*args)
        elif command == "gif":
            make_gif(*args)
        else:
            print("unknown command")

    redraw()
    while True:
        try:
            command()
        except Exception as exc:
            print(exc)


@cli.command()
@cli.argument("output-gif", type=str)
@cli.argument("duration", type=int)
@cli.argument("fps", type=int)
@cli.argument("start", type=float, nargs=4)
@cli.argument("end", type=float, nargs=4)
@cli.argument("-p", "--pixels", type=int, default=1200)
@cli.argument("-n", "--iterations", type=int, default=200)
@cli.argument("-m", "--mode", choices=GRADIENT_MODES.keys(), default="smooth")
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="rainbow")
def gif_zoom(
    output_gif: str,
    duration: int,
    fps: int,
    start: List[float],
    end: List[float],
    pixels: int,
    iterations: int,
    mode: str,
    color: str,
):
    axes_start = Grid.from_floats(*start)
    axes_end = Grid.from_floats(*end)
    width = pixels
    height = round(pixels * axes_start.yx_ratio)

    colorizer = i_values_colorizer(iterations, mode, color)
    frame_count = duration * fps
    frames = []

    for frame_n in range(frame_count + 1):
        start_t = time.time()
        frac = frame_n / frame_count
        print(frac)
        axes = axes_start.interpolate_geometric(axes_end, frame_n / frame_count)
        print(axes)
        grid = generate_complex_coordinates(
            width,
            height,
            axes,
        )
        mandelbrot = MandelbrotSet(grid)
        mandelbrot.iterate(iterations, 5)
        frames.append(mandelbrot.to_image(colorizer))
        print(f"generated frame {frame_n} in {time.time() - start_t:.2f} seconds")

    frames = loop_frames(frames, freeze=fps)

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=1000 / fps,
        loop=0,
    )
    print(f"GIF saved to file://{output_gif}")


@cli.command()
@cli.argument("output-gif", type=str)
@cli.argument("--location", type=float, nargs=4, default=[-2.0, 0.6, -1.2, 1.2])
@cli.argument("-p", "--pixels", type=int, default=2600)
@cli.argument("-n", "--iterations", type=int, default=100)
@cli.argument("-d", "--duration", type=int, default=10)
@cli.argument("-m", "--mode", choices=GRADIENT_MODES.keys(), default="smooth")
@cli.argument("-c", "--color", choices=PALETTES.keys(), default="rainbow")
def gif_iterations(
    output_gif: str,
    location: List[float],
    pixels: int,
    iterations: int,
    duration: int,
    mode: str,
    color: str,
):
    axes = Grid.from_floats(*location)
    width = pixels
    height = round(pixels * axes.yx_ratio)
    grid = generate_complex_coordinates(width, height, axes)
    colorizer = i_values_colorizer(iterations, mode, color)
    mandelbrot = MandelbrotSet(grid)
    frames = []

    for i in range(iterations):
        start = time.time()
        mandelbrot.iterate()
        frames.append(mandelbrot.to_image(colorizer))
        print(f"generated frame {i} in {time.time() - start:.2f} seconds")

    frames = loop_frames(frames, freeze=10)

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=(1000 * duration) // iterations,
        loop=0,
    )
    print(f"GIF saved to file://{output_gif}")


def main():
    np.seterr(all="ignore")
    cli()


if __name__ == "__main__":
    main()
