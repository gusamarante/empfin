from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.1'
DESCRIPTION = 'Empirical Finance Tools'

# Setting up
setup(
    name="empfin",
    version=VERSION,
    author="Gustavo Amarante",
    maintainer="Gustavo Amarante",
    maintainer_email="gustavoca2@insper.edu.br",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit_learn",
        "scipy",
        "seaborn",
        "setuptools",
        "statsmodels",
        "tqdm",
    ],
    keywords=[
        'asset pricing',
        'empirical asset pricing',
        'empirical finance',
        'factor models',
        'finance',
        'risk premia',
    ],
)