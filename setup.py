from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1'
DESCRIPTION = 'Empirical Finance Tools'

# Setting up
setup(
    name="empfin",
    version=VERSION,
    author="Gustavo Amarante",
    maintainer="Gustavo Amarante",
    maintainer_email="developer@dsgepy.com",  # TODO change this
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "scipy",
        "matplotlib",
        "numpy",
        "pandas",
    ],
    keywords=[
        'asset pricing',
        'empirical asset pricing',
        'factor models',
        'risk premia',
    ],
)