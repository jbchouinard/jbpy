def fiter(f, init):
    acc = init
    while True:
        yield acc
        acc = f(acc)


def nth(itr, n):
    """Get nth item from an iterator."""
    itr = iter(itr)
    try:
        v = next(itr)
        for _ in range(n):
            v = next(itr)
        return v
    except StopIteration:
        raise IndexError("index out of range")


def slice_max_idx(s: slice):
    """
    Calculates the maximum possible index contained in a slice,
    for an unwknown list size.

    Returns -1 if the maximum possible index is the end of the list.
    """
    if s.step is None or s.step > 0:
        start, end = s.start, s.stop
    elif s.step < 0:
        start, end = s.stop, s.start
    else:
        raise ValueError("slice step cannot be zero")

    if end is None or end < 0:
        return -1

    if start is not None and start >= end:
        return 0

    return end - 1


class lazylist:
    def __init__(self, itr):
        self.itr = iter(itr)
        self.values = []

    def _get_up_to_idx(self, idx):
        if idx == -1:
            while True:
                try:
                    self.values.append(next(self.itr))
                except StopIteration:
                    break

        else:
            for _ in range(len(self.values), idx + 1):
                try:
                    self.values.append(next(self.itr))
                except StopIteration:
                    break

    def __len__(self):
        self._get_up_to_idx(-1)
        return len(self.values)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            max_idx = idx
        elif isinstance(idx, slice):
            max_idx = slice_max_idx(idx)
        else:
            raise TypeError("cannot index by this type")

        self._get_up_to_idx(max_idx)
        return self.values[idx]
