import json
from abc import ABC, abstractmethod
from os import PathLike

from typing import List, Literal, get_args, Dict, Any, Union

from pydantic import BaseModel

from arc.register_builders import RegisterBuilder

TypePhonemeFeatureLabels = Literal[
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr", "lab",
    "hi", "lo", "back", "round", "tense", "long"
]
PHONEME_FEATURE_LABELS = list(get_args(TypePhonemeFeatureLabels))

LABELS_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
LABELS_V = ['back', 'hi', 'lo', 'lab', 'tense', 'long']
N_FEAT = len(LABELS_C) + len(LABELS_V)  # 14

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
    
    def flatten(self):
        register_builder = RegisterBuilder()
        for sub_element in self.get_elements():
            register_builder.add_item(sub_element)
        return register_builder.build()


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


class Stream(Element, BaseModel):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]

    def get_elements(self):
        return self.syllables
        
    def __str__(self):
        return "|".join(syllable.id for syllable in self)
    
    def save(self, path: Union[str, PathLike] = None):
        if path is None:
            path = f"stream.json"

        if isinstance(path, str) and not path.endswith(".json"):
            path = path + ".json"

        with open(path, "w", encoding='utf-8') as file:
            json.dump(self, file,
                      default=lambda o: o.model_dump(), sort_keys=False, ensure_ascii=False)