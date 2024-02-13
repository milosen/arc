import json
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from os import PathLike
from typing import List, Dict, Any, TypeVar, Union, Literal, get_args

from pydantic import BaseModel

TypePhonemeFeatureLabels = Literal[
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr", "lab",
    "hi", "lo", "back", "round", "tense", "long"
]

PHONEME_FEATURE_LABELS = list(get_args(TypePhonemeFeatureLabels))


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


class Phoneme(Element, BaseModel):
    id: str
    info: Dict[str, Any]

    def get_elements(self):
        return []

    def get_feature_symbol(self, label: TypePhonemeFeatureLabels):
        return self.info["features"][PHONEME_FEATURE_LABELS.index(label)]

    def get_binary_feature(self, label: TypePhonemeFeatureLabels):
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


SyllableStream = Word


class Lexicon(BaseModel, Element):
    id: str
    words: List[Word]
    info: Dict[str, Any]

    def get_elements(self):
        return self.words


TypeRegister = TypeVar("TypeRegister", bound="Register")


class Register(OrderedDict):
    MAX_PRINT_ELEMENTS = 10
    INFO_KEY = "_info"

    def __init__(self, other=(), /, **kwargs):
        if self.INFO_KEY in kwargs:
            self.info = kwargs[self.INFO_KEY]
            del kwargs[self.INFO_KEY]
        else:
            self.info = {}

        super().__init__(other, **kwargs)

    def __contains__(self, item: Union[str, Element]):
        if isinstance(item, str):
            return item in self.keys()
        elif isinstance(item, Element):
            return item.id in self.keys()
        else:
            raise ValueError("item type unknown")

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value: Dict):
        self._info = value

    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return list(self.values())[item]

        return super().__getitem__(item)

    def __str__(self):
        li = list(self.keys())
        n_elements = len(li)

        s = "|".join(li[:self.MAX_PRINT_ELEMENTS])

        if n_elements > self.MAX_PRINT_ELEMENTS:
            s += "|..."

        s += f" ({n_elements} elements total)"

        return s

    def append(self, obj: Element):
        self[str(obj)] = obj

    def get_subset(self, size: int) -> TypeRegister:
        """Create a new Register as a random subset of this one"""
        if size >= len(self):
            return self

        keys = set()

        for _ in range(size):
            keys.add(random.choice(list(self.keys() - keys)))

        return Register({key: self[key] for key in keys}, _info=self.info)

    def get_self_with_info_key(self):
        d = copy(self)
        d.update({self.INFO_KEY: self.info})
        return d

    def to_json(self):
        return json.dumps(self.get_self_with_info_key(), default=lambda o: o.model_dump(),
                          sort_keys=False, ensure_ascii=False)

    def save(self, path: Union[str, PathLike] = None):
        if path is None:
            path = f"{self[0].__class__.__name__.lower()}s.json"

        if isinstance(path, str) and not path.endswith(".json"):
            path = path + ".json"

        with open(path, "w") as file:
            json.dump(self.get_self_with_info_key(), file,
                      default=lambda o: o.model_dump(), sort_keys=False, ensure_ascii=False)

    def intersection(
            self,
            other: TypeRegister
    ) -> TypeRegister:
        """
        Select Elements that are in both corpora and merge the data
        :param other:
        :return:
        """

        def merge_infos(element_1, element_2):
            element_new = copy(element_1)
            element_new.info.update(element_2.info)
            return element_new

        new_corpus_info = copy(self.info)
        new_corpus_info.update(other.info)

        return Register({
            key: merge_infos(element, other[key]) for key, element in self.items() if key in other
        }, _info=new_corpus_info)
