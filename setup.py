import io

from setuptools import setup, find_packages

__version__ = '0.1.1.dev1'

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
        "numpy>=1.24.4",
        "scipy>=1.10.1",
        "rich>=13.5.3",
        "tqdm>=4.66.1",
        "pydantic>=2.5.2",
        "pytest>=7.4.3"
    ],
    extras_require={},
    include_package_data=True,
    package_data={'arc': ['data/*']},
)
