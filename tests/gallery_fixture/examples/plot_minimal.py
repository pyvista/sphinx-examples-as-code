"""A minimal gallery example
==========================

This example references :func:`sum` and includes a note.

.. note::

   This is a note written inside the gallery example itself.
"""

# %%
# Some prose in its own cell, before the code.

from __future__ import annotations

x = 1
y = 2
print(x + y)

# %%
# A headed cell
# ~~~~~~~~~~~~~
# This heading turns the rest of the script into a *sibling* section at the
# document level (not nested inside the page's own section) -- regression
# coverage for the code here silently being dropped.

z = 3
print(z)
