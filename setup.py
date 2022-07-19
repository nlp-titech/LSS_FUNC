import io
import os

from setuptools import setup

# Package meta-data.
NAME = "LSS_FUNC"
DESCRIPTION = "Codes for TOD95."
URL = "https://github.com/meshidenn/"
EMAIL = "hiroki.iida@nlp.c.titech.ac.jp"
AUTHOR = "Hiroki Iida"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.0.1"

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


def _requires_from_file(filename):
    open_file = os.path.join(here, filename)
    return open(open_file).read().splitlines()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["lss_func"],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=_requires_from_file("requirements.txt"),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
)
