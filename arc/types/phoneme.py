from typing import Literal, get_args, TypeVar, Dict, Any

from pydantic import BaseModel

from arc.types.base_types import Element

TypePhonemeFeatureLabels = Literal[
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr", "lab",
    "hi", "lo", "back", "round", "tense", "long"
]
PHONEME_FEATURE_LABELS = list(get_args(TypePhonemeFeatureLabels))
PhonemeType = TypeVar("PhonemeType", bound="Phoneme")


class Phoneme(Element, BaseModel):
    id: str
    info: Dict[str, Any]

    def get_elements(self):
        return []

    def get_feature_symbol(self, label: TypePhonemeFeatureLabels):
        return self.info["features"][PHONEME_FEATURE_LABELS.index(label)]

    def get_binary_feature(self, label: TypePhonemeFeatureLabels):
        return self.get_feature_symbol(label) == "+"
