import itertools
from typing import TypeVar, List, Dict, Any

import numpy as np
from pydantic import BaseModel

from arc.controls.common import get_oscillation_patterns
from arc.core.base_types import Register, Element
from arc.core.syllable import Syllable


def check_syll_feature_overlap(syllables):
    all_feats = [feat for syll in syllables for phon_feats in syll.info["phonotactic_features"] for feat in phon_feats]
    return len(all_feats) == len(set(all_feats))


def generate_subset_syllables(syllables, lookback_syllables):
    subset = []
    for new_syll in syllables:
        syll_test_set = lookback_syllables + [new_syll]
        if check_syll_feature_overlap(syll_test_set):
            subset.append(new_syll)

    return subset


class Word(Element, BaseModel):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]

    def get_elements(self):
        return self.syllables


def word_overlap_matrix(words: Register[str, Word], lag_of_interest: int = 1):
    n_words = len(words)
    n_sylls_per_word = len(words[0].syllables)

    oscillation_patterns = get_oscillation_patterns(lag=(n_sylls_per_word*lag_of_interest))

    overlap = np.zeros([n_words, n_words], dtype=int)
    for i1, i2 in list(itertools.product(range(n_words), range(n_words))):
        word_pair_features = [f1 + f2 for f1, f2 in zip(words[i1].info["binary_features"],
                                                        words[i2].info["binary_features"])]

        matches = 0
        for word_pair_feature in word_pair_features:
            if word_pair_feature in oscillation_patterns:
                matches += 1

        overlap[i1, i2] = matches

    return overlap


WordType = TypeVar("WordType", bound="Word")
