import itertools
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Any, Generator, List, Tuple, Dict, Set

from pydantic import BaseModel, ValidationError, PositiveInt

# lexica[words[syllables[phonemes]]]
example_word = {
    "id": "ka:fu:ry",
    "info": {},
    "syllables": [
        {
            "id": "ka:",
            "info": {},
            "phonemes": [
                {
                    "id": "k",
                    "info": {},
                    "order": None,
                    "features": None
                },
                {
                    "id": "a:",
                    "info": {},
                    "order": None,
                    "features": None
                }
            ]
        },
        {
            "id": "fu:",
            "info": {},
            "phonemes": [
                {
                    "id": "f",
                    "info": {},
                    "order": [],
                    "features": []
                },
                {
                    "id": "u:",
                    "info": {},
                    "order": [],
                    "features": []
                }
            ]
        },
        {
            "id": "ry:",
            "info": {},
            "phonemes": [
                {
                    "id": "r",
                    "info": {},
                    "order": [],
                    "features": []
                },
                {
                    "id": "y:",
                    "info": {},
                    "order": [],
                    "features": []
                }
            ]
        },
    ]
}


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

    def __str__(self):
        return self.id

    def decompose(self):
        return self.phonemes


class Word(BaseModel, BaseDictARC):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.syllables


class Lexicon(BaseModel, BaseDictARC):
    id: str
    words: List[Word]
    info: Dict[str, Any]

    def __str__(self):
        return [w.id for w in self.words].__str__()

    def decompose(self):
        return self.words


if __name__ == '__main__':
    try:
        syllables = Syllable()
        word = Word(**example_word)
        print(word.as_dict())
        lex = Lexicon(**{"id": "L1", "words": [example_word, example_word], "info": {}})
        print(lex.as_dict())
    except ValidationError as e:
        print(e.errors())
