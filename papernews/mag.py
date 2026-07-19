"""Mag notation: numbers as their base-10 magnitude (magworld.pw).

A number is written as its common logarithm behind a ``^`` sigil: ``^3``
is 1000, ``^3.4`` is ~2500, ``^-2`` is 1/100, Avogadro is ``^23.8``. The
magnitude — the part scientific notation tucks away in small type at the
end — becomes the whole number, and the mantissa clutter disappears.

The "clever rounding": one decimal of magnitude is a factor of ~1.26
(26%), which is exactly the resolution at which order-of-magnitude
thinking stays honest. Everything rounds to that grid, and magnitudes
that land on an integer drop the decimal entirely so the round powers of
ten read cleanly. Multiplication becomes addition; 0 is human scale; the
world runs roughly ^-10 to ^+10 in every dimension.
"""

from __future__ import annotations

import math

SIGIL = "^"


def mag(x: float) -> float | None:
    """The magnitude (log10) of x, on the 0.1 grid. None if undefined."""
    if not math.isfinite(x) or x <= 0:
        return None
    return round(math.log10(x), 1)


def format_mag(x: float) -> str:
    """Render x in mag notation: ^3, ^3.4, ^-2.7; -^3 for negatives.

    Zero stays "0" (its magnitude is undefined, and the reader's "nothing"
    beats a sigil). Integral magnitudes drop the decimal.
    """
    if x == 0:
        return "0"
    if x < 0:
        return "-" + format_mag(-x)
    m = mag(x)
    if m is None:
        return str(x)
    if m == int(m):
        return f"{SIGIL}{int(m)}"
    return f"{SIGIL}{m:.1f}"
