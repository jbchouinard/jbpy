from setuptools import setup, find_packages

setup(
    name="jb",
    version="0.1.0",
    packages=find_packages(include=["jb","jb.*"]),
    entry_points={
        "console_scripts": ["jbm=jb.fractals.mandelbrot:main"],
    },
)
