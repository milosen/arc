import io

from setuptools import setup, find_packages

__version__ = '1.0'

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='arc',
    version=__version__,
    author='Lorenzo Titone, Nikola Milosevic',
    author_email='milose.nik@gmail.com',
    description='ARC: A tool for creating artificial languages with rhythmicity control',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
        "pydantic",
        "pytest",
        "pingouin",
        "matplotlib",
        "Click",
    ],
    include_package_data=True,
    package_data={'arc': ['data/*', 'data/example_corpus/*']},
    entry_points={
        'console_scripts': [
            'arc = arc.cli:cli',
        ],
    },
)
