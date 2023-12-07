import logging
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Any, NewType

from pydantic import BaseModel, PositiveInt, NonNegativeInt

from arc.definitions import PHONEME_FEATURE_LABELS


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

    def __iter__(self):
        return iter(self.decompose())

    @abstractmethod
    def decompose(self):
        pass


class CollectionARC(OrderedDict):
    def __contains__(self, item):
        return item in self.keys()

    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return list(self.values())[item]
        return super().__getitem__(item)

    def append(self, obj: BaseDictARC):
        self[str(obj)] = obj

    def __str__(self):
        li = list(self.keys())
        return "|".join(li[:10]) + "|..." + f" ({len(li)} elements total)"

    def sample(self, n):
        if n >= len(self):
            logging.warning("Sampling more or equal then the collection size just gives you back the collection itself"
                            "because the elements are unique.")
            return self

        keys = set()

        for _ in range(n):
            keys.add(random.choice(list(self.keys() - keys)))

        return CollectionARC({key: self[key] for key in keys})


class Phoneme(BaseDictARC, BaseModel):
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

    def get_feature(self, label):
        return self.features[PHONEME_FEATURE_LABELS.index(label)]


class Syllable(BaseDictARC, BaseModel):
    id: str
    phonemes: List[Phoneme]
    info: Dict[str, Any]
    binary_features: List[NonNegativeInt]
    phonotactic_features: List[List[str]]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.phonemes


class Word(BaseDictARC, BaseModel):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]
    binary_features: List[List[NonNegativeInt]]

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


WordType = NewType("WordType", Word)
LexiconType = NewType("LexiconType", Lexicon)
SyllableType = NewType("SyllableType", Syllable)
PhonemeType = NewType("PhonemeType", Phoneme)
CollectionARCType = NewType("CollectionARCType", CollectionARC)
