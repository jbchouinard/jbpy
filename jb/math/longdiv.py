import itertools
import math
from typing import Optional


class DecimalNumber:
    def __init__(
        self,
        whole: int,
        decimals: int,
        repeating_start: Optional[int] = None,
        truncated: bool = False,
    ):
        self.whole = whole
        self.decimals = decimals
        self.repeating_start = repeating_start
        self.truncated = truncated

    def iter_decimals(self):
        digits = str(self.decimals)[1:]
        if self.repeating_start is not None:
            yield from digits[: self.repeating_start]
            yield from itertools.cycle(digits[self.repeating_start :])
        else:
            yield from digits

    def period_length(self):
        if self.repeating_start is None:
            return 0
        else:
            return len(str(self.decimals)) - 1 - self.repeating_start

    def to_int(self):
        return self.whole

    def to_float(self):
        decimals = "".join(itertools.islice(self.iter_decimals(), 0, 20))
        return float(f"{self.whole}.{decimals}")

    def __str__(self):
        parts = []

        if self.truncated:
            parts.append("~")

        parts.append(str(self.whole))
        parts.append(".")

        digits = str(self.decimals)[1:]
        if self.repeating_start is not None:
            parts.append(digits[: self.repeating_start])
            parts.append("(")
            parts.append(digits[self.repeating_start :])
            parts.append(")")
        else:
            parts.append(digits)

        return "".join(parts)


def long_division(dividend: int, divisor: int, max_precision: int):
    seen_remainders = {}  # detects repeating decimals
    whole = 0
    decimals = 1
    remainder = dividend
    decimal_place = 0

    while True:
        if remainder == 0:
            return DecimalNumber(whole, decimals)

        if remainder in seen_remainders:
            return DecimalNumber(
                whole, decimals, repeating_start=seen_remainders[remainder]
            )
        else:
            seen_remainders[remainder] = decimal_place

        if max_precision and (decimal_place >= max_precision):
            return DecimalNumber(whole, decimals, truncated=True)

        if remainder < divisor:
            decimal_place += 1
            remainder *= 10

        quotient = remainder // divisor
        remainder = remainder % divisor

        if decimal_place == 0:
            whole = quotient
        else:
            decimals = decimals * 10 + quotient


class Fraction:
    def __init__(self, numerator: int, denominator: int):
        self.n = numerator
        self.d = denominator

    def __eq__(self, other):
        return self.simplified() == other.simplified()

    def __str__(self):
        return f"{self.n}/{self.d}"

    def simplified(self):
        gcd = math.gcd(self.n, self.d)
        return Fraction(self.n // gcd, self.d // gcd)

    def to_decimal(self, precision: int) -> DecimalNumber:
        return long_division(self.n, self.d, precision)

    def to_int(self):
        return self.to_decimal(1).to_int()

    def to_float(self):
        return self.to_decimal(20).to_float()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dividend", type=int)
    parser.add_argument("divisor", type=int)
    parser.add_argument("-p", "--precision", type=int, default=20)
    parser.add_argument("-l", "--period-length", action="store_true")
    args = parser.parse_args()

    frac = Fraction(args.dividend, args.divisor)
    parts = [str(frac)]
    sfrac = frac.simplified()
    if sfrac.n != frac.n:
        parts.append(str(sfrac))

    if args.period_length:
        n = sfrac.to_decimal(precision=0)
        parts.append(str(n.period_length()))
    else:
        n = sfrac.to_decimal(precision=args.precision)
        parts.append(str(n))

    print("\n".join(parts))


if __name__ == "__main__":
    main()
