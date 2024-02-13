import json
import warnings
from os import PathLike
from typing import Union, Type

from tqdm import TqdmExperimentalWarning

from .io import load_phonemes, load_syllables, load_words, load_lexicons
from .types import Phoneme, Syllable, Word, Lexicon, Register, TypeRegister, Element

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
