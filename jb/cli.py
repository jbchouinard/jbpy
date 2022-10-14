import argparse
from collections import defaultdict
from typing import Callable, Optional, TypeVar


T = TypeVar("T")


class Command:
    def __init__(self, name, func, parser, subparser):
        self.name = name
        self.func = func
        self.parser = parser
        self.subparser = subparser

    def __call__(self, command, **kwargs):
        self.func(**kwargs)


class CLI:
    def __init__(self, **kwargs):
        self.parser = argparse.ArgumentParser(**kwargs)
        self.subparsers = self.parser.add_subparsers(dest="command", required=True)
        self.commands = {}
        self.arguments = defaultdict(list)

    def command(self, *args, **kwargs):
        def decorator(func):
            name = func.__name__.replace("_", "-")
            subparser = self.subparsers.add_parser(name, *args, **kwargs)
            for arg_args, arg_kwargs in self.arguments[name]:
                subparser.add_argument(*arg_args, **arg_kwargs)

            command = Command(name, func, self.parser, subparser)
            self.commands[name] = command
            return func

        return decorator

    def argument(self, *args, **kwargs):
        def decorator(func):
            name = func.__name__.replace("_", "-")
            self.arguments[name].append((args, kwargs))
            return func

        return decorator

    def __call__(self):
        args = self.parser.parse_args()
        self.commands[args.command](**vars(args))


def get_input(prompt, vdef: Optional[str] = None, vtype: Callable[[str], T] = str) -> T:
    if vdef is not None:
        prompt = f"{prompt} ({vdef}): "
    else:
        prompt = f"{prompt}: "
    while True:
        try:
            raw = input(prompt)
            if not raw:
                if vdef is not None:
                    return vtype(vdef)
                else:
                    print("enter a value")
                    continue
            return vtype(raw)
        except Exception as exc:
            print(exc)


if __name__ == "__main__":
    cli = CLI()

    @cli.command()
    @cli.argument("foo")
    def test_a(foo):
        print(foo)

    @cli.command()
    @cli.argument("bar")
    def test_b(bar):
        print(bar)

    cli()
