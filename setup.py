from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "1.1.3"
DESCRIPTION = "Construtor de regressão genérico."
LONG_DESCRIPTION = ""

# Setting up
setup(
    name="free_regression",
    version=VERSION,
    author=["Ian dos Anjos Melo Aguiar", "Arthur Magalhões", "João Roberto", "Henrique de Souza"],
    author_email="<iannaianjos@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["matplotlib", "numpy"],
    keywords=["python", "regression"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
