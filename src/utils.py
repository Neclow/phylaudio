import subprocess
import sys
from itertools import islice

# Test if the current platform is Windows or not
IS_WINDOWS = sys.platform.startswith("win")


def _run_command(command, **kwargs):
    """Basic shell command runner"""
    if IS_WINDOWS:
        command = f"wsl {command}"

    subprocess_args = {
        "capture_output": kwargs.get("capture_output", True),
        "shell": kwargs.get("shell", not IS_WINDOWS),
    }

    try:
        output = subprocess.run(command, check=True, **subprocess_args)
    except subprocess.CalledProcessError as _:
        # pylint: disable=subprocess-run-check
        output = subprocess.run(command, **subprocess_args)
        # pylint: enable=subprocess-run-check

        raise RuntimeError(output) from _


def _count_file_lines(fname):
    """
    Count the number of lines in a file

    Source: https://stackoverflow.com/a/68385697/17687514

    Parameters
    ----------
    fname : str
        File name
    """

    def _make_gen(reader):
        while True:
            b = reader(2**16)
            if not b:
                break
            yield b

    with open(fname, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count


# TODO: update to Python 3.12
def batched(iterable, n, *, strict=False):
    """
    Batch data from the iterable into tuples of length n.
    The last batch may be shorter than n.

    From: https://docs.python.org/3/library/itertools.html#itertools.batched

    Parameters
    ----------
    iterable : Iterable
        An iterable
    n : int
        Batch size
    strict : bool, optional
        If True, raises a ValueError if the final batch is shorter than n, by default False

    Yields
    ------
    tuple
        Batch of size <= n
    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch
