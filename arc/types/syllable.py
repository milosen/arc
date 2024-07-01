from typing import List, Literal, Optional, Union, TypeVar, Dict, Any

from pydantic import BaseModel

from arc.types.phoneme import Phoneme
from arc.types.base_types import Element

LABELS_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
LABELS_V = ['back', 'hi', 'lo', 'lab', 'tense', 'long']
N_FEAT = len(LABELS_C) + len(LABELS_V)  # 14

SyllableType = TypeVar("SyllableType", bound="Syllable")


class Syllable(Element, BaseModel):
    id: str
    phonemes: List[Phoneme]
    info: Dict[str, Any]

    def get_elements(self):
        return self.phonemes