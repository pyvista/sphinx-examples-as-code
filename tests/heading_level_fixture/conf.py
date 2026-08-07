"""Sphinx config for testing gallery sibling-section heading levels."""

from __future__ import annotations

extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx_examples_as_code',
]

root_doc = 'index'
project = 'heading_level_fixture'
exclude_patterns = ['_build', 'examples/GALLERY_HEADER.rst']

sphinx_gallery_conf = {
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
    'image_scrapers': (),
    'plot_gallery': 'True',
}

sphinx_examples_as_code_conf = {
    'gallery_downloads': True,
}
