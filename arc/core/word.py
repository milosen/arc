import itertools
import logging
import random
from copy import copy
from typing import TypeVar, List, Dict, Any

import numpy as np
from pydantic import BaseModel
from tqdm.rich import tqdm

from arc.core.syllable import Syllable
from arc.core.base_types import Register, Element, RegisterType

from arc.controls.common import get_oscillation_patterns


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


def make_words(syllables: RegisterType,
               n_sylls=3,
               n_look_back=2,
               n_words=10_000,
               max_tries=100_000,
               progress_bar: bool = False,
               ) -> RegisterType:
    words = {}

    iter_tries = range(max_tries)

    if progress_bar:
        pbar = tqdm(total=n_words)

    for _ in iter_tries:
        sylls = []
        for _ in range(n_sylls):
            sub = generate_subset_syllables(syllables, sylls[-n_look_back:])
            sub = list(filter(lambda x: x.id not in sylls, sub))
            if sub:
                syll = random.sample(sub, 1)[0]
                sylls.append(syll)

        if len(sylls) == n_sylls:
            word_id = "".join(s.id for s in sylls)
            if word_id not in words:
                word_features = list(list(tup) for tup in zip(*[s.info["binary_features"] for s in sylls]))
                words[word_id] = Word(id=word_id, info={"binary_features": word_features}, syllables=sylls)
                if progress_bar:
                    pbar.update(1)

        if len(words) == n_words:
            logging.info(f"Done: Found {n_words} words.")
            break

    words_register = Register(words)
    words_register.info = copy(syllables.info)

    return words_register


class Word(Element, BaseModel):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]

    def get_elements(self):
        return self.syllables


def word_overlap_matrix(words: Register[str, Word]):
    n_words = len(words)
    n_sylls_per_word = len(words[0].syllables)

    oscillation_patterns = get_oscillation_patterns(lag=n_sylls_per_word)

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
