"""Sphinx config for testing the sphinx-gallery integration."""

from __future__ import annotations

extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx_examples_as_code',
]

root_doc = 'index'
project = 'gallery_fixture'
# GALLERY_HEADER.rst is consumed by sphinx-gallery itself (folded into the
# generated auto_examples/index page), not a standalone document -- exclude
# it so it isn't also built (and warned about) on its own.
exclude_patterns = ['_build', 'examples/GALLERY_HEADER.rst']

sphinx_gallery_conf = {
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
    # No matplotlib/pyvista scraper needed for these tests -- keeps the
    # fixture light and avoids a plotting-backend dependency.
    'image_scrapers': (),
    'plot_gallery': 'True',
}
