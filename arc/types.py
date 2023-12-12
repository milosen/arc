import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from os import PathLike
from typing import List, Dict, Any, NewType, TypeVar, Union, Type

from pydantic import BaseModel, PositiveInt, NonNegativeInt

from arc.definitions import PHONEME_FEATURE_LABELS, RESULTS_DEFAULT_PATH
from typing import TypeVar, Generic, Iterable

T = TypeVar('T')
S = TypeVar('S')


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


class CollectionARC(OrderedDict, Generic[S, T]):
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

    def save(self, path: Union[str, PathLike] = None):
        if path is None:
            path = RESULTS_DEFAULT_PATH / f"arc_{self[0].__class__.__name__}s.json"

        if isinstance(path, str) and not path.endswith(".json"):
            path = path + ".json"

        with open(path, "w") as file:
            json.dump(self, file, default=lambda o: o.model_dump(), sort_keys=True, ensure_ascii=False)


def from_json(path: Union[str, PathLike], arc_type: Type = T) -> CollectionARC[str, T]:
    with open(path, "r") as file:
        d = json.load(file)

    return CollectionARC({k: arc_type(**v) for k, v in d.items()})
