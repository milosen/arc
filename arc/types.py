from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass(frozen=True)
class Phoneme:
    phon: str
    order: int


@dataclass(frozen=True)
class Syllable:
    syll: str
    freq: int
    prob: float
    p_unif: float


@dataclass(frozen=True)
class Ngram:
    ngram: str
    freq: float
    p_unif: float


PhonemesList = List[Phoneme]
SyllablesList = List[Syllable]
NgramsList = List[Ngram]
Word = Tuple[Syllable, ...]
WordsList = List[Word]
Lexicon = Set[Word]


@dataclass
class BinaryFeatures:
    labels: List[str]
    labels_c: List[str]
    labels_v: List[str]
    phons: List[str]
    numbs: List[str]
    consonants: List[str]
    long_vowels: List[str]
    n_features: int

    def print_value(self, value):
        s = ""
        if isinstance(value, list):
            s += "["
            for v in value[:3]:
                s += str(self.print_value(v))
                s += ", "
            s += "...]"
        else:
            s += str(value)
        return s

    def __str__(self):
        s = ""
        for key, value in self.__dict__.items():
            s += key
            s += ": "
            s += self.print_value(value)
            s += ", "
        s = s[:-2]
        return f"BinaryFeatures({s})"
