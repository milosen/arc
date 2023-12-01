from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

from pydantic import BaseModel, PositiveInt, NonNegativeInt


class BaseDictARC(ABC):
    def full_str(self):
        """recursive string representation"""
        full_str = "["
        for entry in self.decompose():
            full_str += entry.full_str()
            full_str += ", "
        full_str = full_str[:-2]
        full_str += "]"
        return f"{self.__str__()} -> {full_str}"

    def as_dict(self):
        """recursive dict representation"""
        full_dict = {}
        full_list = []
        for entry in self.decompose():
            next_level = entry.as_dict()
            if isinstance(next_level, str):
                full_list.append(next_level)
            else:
                full_dict.update(**next_level)

        return {self.__str__(): full_dict or full_list}

    def get_vals(self, key: str) -> List:
        return [entry.info[key] for entry in self.decompose()]

    def __getitem__(self, item):
        return self.decompose()[item]

    @abstractmethod
    def decompose(self):
        pass


class Phoneme(BaseModel, BaseDictARC):
    id: str
    info: Dict[str, Any]
    order: List[PositiveInt]
    features: List[str]

    def __str__(self):
        return self.id

    def full_str(self):
        return self.__str__()

    def decompose(self):
        return []

    def as_dict(self):
        return self.id


class Syllable(BaseModel, BaseDictARC):
    id: str
    phonemes: List[Phoneme]
    info: Dict[str, Any]
    features: List[NonNegativeInt]
    custom_features: List[List[str]]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.phonemes


class Word(BaseModel, BaseDictARC):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]
    features: List[List[NonNegativeInt]]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.syllables


class Lexicon(BaseModel, BaseDictARC):
    id: str
    words: List[Word]
    info: Dict[str, Any]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.words


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
