import dataclasses
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import List, Dict, Any, NewType, TypeVar, Union, Type, Literal

from pydantic import BaseModel, PositiveInt, NonNegativeInt

from arc.definitions import PHONEME_FEATURE_LABELS, RESULTS_DEFAULT_PATH
from typing import TypeVar, Generic

T = TypeVar('T')
S = TypeVar('S')


class Element(ABC):
    id: str
    info: Dict[str, Any]

    def __getitem__(self, item):
        return self.get_elements()[item]

    def __iter__(self):
        return iter(self.get_elements())

    def __str__(self):
        return self.id

    @abstractmethod
    def get_elements(self):
        pass

    @classmethod
    def from_json(cls, path: Union[str, PathLike]):
        with open(path, "r") as file:
            d = json.load(file)

        return Register({k: cls.__init__(**v) for k, v in d.items()})


PhFeatureLabels = Literal["syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid", "voi", "sg", "cg", "ant",
                          "cor", "distr", "lab", "hi", "lo", "back", "round", "tense", "long"]


class Phoneme(Element, BaseModel):
    id: str
    info: Dict[str, Any]

    def get_elements(self):
        return []

    def get_feature_symbol(self, label: PhFeatureLabels):
        return self.info["features"][PHONEME_FEATURE_LABELS.index(label)]

    def get_binary_feature(self, label: PhFeatureLabels):
        return self.get_feature_symbol(label) == "+"


class Syllable(Element, BaseModel):
    id: str
    phonemes: List[Phoneme]
    info: Dict[str, Any]

    def get_elements(self):
        return self.phonemes


class Word(Element, BaseModel):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]

    def get_elements(self):
        return self.syllables


class Lexicon(BaseModel, Element):
    id: str
    words: List[Word]
    info: Dict[str, Any]

    def get_elements(self):
        return self.words


TypeRegister = TypeVar("TypeRegister")


class Register(OrderedDict, Generic[S, T]):
    MAX_PRINT_ELEMENTS = 10

    def __contains__(self, item: Union[str, Element]):
        if isinstance(item, str):
            return item in self.keys()
        elif isinstance(item, Element):
            return item.id in self.keys()
        else:
            raise ValueError("item type unknown")

    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return list(self.values())[item]

        return super().__getitem__(item)

    def __str__(self):
        li = list(self.keys())
        return "|".join(li[:self.MAX_PRINT_ELEMENTS]) + "|..." + f" ({len(li)} elements total)"

    def append(self, obj: Element):
        self[str(obj)] = obj

    def get_subset(self, size: int):
        """Create a new Register as a random subset of this one"""
        if size >= len(self):
            return self

        keys = set()

        for _ in range(size):
            keys.add(random.choice(list(self.keys() - keys)))

        return Register({key: self[key] for key in keys})

    def save(self, path: Union[str, PathLike] = None):
        if path is None:
            path = RESULTS_DEFAULT_PATH / f"arc_{self[0].__class__.__name__}s.json"

        if isinstance(path, str) and not path.endswith(".json"):
            path = path + ".json"

        with open(path, "w") as file:
            json.dump(self, file, default=lambda o: o.model_dump(), sort_keys=True, ensure_ascii=False)


def from_json(path: Union[str, PathLike], arc_type: Type = T) -> Register[str, T]:
    with open(path, "r") as file:
        d = json.load(file)

    return Register({k: arc_type(**v) for k, v in d.items()})
