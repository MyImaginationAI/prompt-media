"""Setup file for prompt-media package."""
from setuptools import setup, find_packages

setup(
    name="prompt-media",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jinja2",
        "pyyaml",
    ],
)
