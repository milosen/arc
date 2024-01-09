import warnings
from os import PathLike
from typing import Union

from tqdm import TqdmExperimentalWarning

from arc.definitions import RESULTS_DEFAULT_PATH, PHONEMES_DEFAULT_PATH
from arc.io import read_phonemes_csv
from arc.types import from_json, Phoneme, Syllable, Word, Lexicon

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def load_default_phonemes():
    return read_phonemes_csv()


def load_phonemes(path: Union[str, PathLike]):
    return from_json(path, Phoneme)


def load_syllables(path: Union[str, PathLike]):
    return from_json(path, Syllable)


def load_words(path: Union[str, PathLike]):
    return from_json(path, Word)


def load_lexicons(path: Union[str, PathLike]):
    return from_json(path, Lexicon)
