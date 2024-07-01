from copy import copy
import itertools
import logging
import random
from typing import TypeVar, List, Dict, Any

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from arc.types.base_types import Register, Element, RegisterType
from arc.types.syllable import Syllable, SyllableType
from arc.types.word import Word, WordType

from arc.controls.common import get_oscillation_patterns
from arc.controls.filter import filter_common_phoneme_words, filter_gram_stats


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


def word_overlap_matrix(words: Register[str, Word], lag_of_interest: int = 1):
    n_words = len(words)
    n_sylls_per_word = len(words[0].syllables)

    oscillation_patterns = get_oscillation_patterns(lag=(n_sylls_per_word*lag_of_interest))

    overlap = np.zeros([n_words, n_words], dtype=int)
    for i1, i2 in list(itertools.product(range(n_words), range(n_words))):
        word_pair_features = [f1 + f2 for f1, f2 in zip(words[i1].info["binary_features"],
                                                        words[i2].info["binary_features"])]

        for word_pair_feature in word_pair_features:
            if word_pair_feature in oscillation_patterns:
                overlap[i1, i2] += 1

    return overlap


def make_words(syllables: RegisterType,
               num_syllables=3,
               bigram_control=True,
               bigram_alpha=None,
               trigram_control=True,
               trigram_alpha=None,
               positional_control=True,
               phonotactic_control=True,
               n_look_back=2,
               n_words=10_000,
               max_tries=100_000,
               progress_bar: bool = True,
               ) -> RegisterType:
    words = {}

    iter_tries = range(max_tries)

    if progress_bar:
        pbar = tqdm(total=n_words)

    for _ in iter_tries:
        sylls = []
        for _ in range(num_syllables):
            sub = list(filter(lambda x: x.id not in sylls, syllables))
            if phonotactic_control:
                sub = generate_subset_syllables(sub, sylls[-n_look_back:])
            if sub:
                new_rand_valid_syllable = random.sample(sub, 1)[0]
                sylls.append(new_rand_valid_syllable)

        if len(sylls) == num_syllables:
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

    words_register = filter_gram_stats(
        words_register,
        bigram_control=bigram_control,
        trigram_control=trigram_control,
        p_val_uniform_bigrams=bigram_alpha,
        p_val_uniform_trigrams=trigram_alpha
    )

    if positional_control:
        print("positional control...")
        words_register = filter_common_phoneme_words(words_register)

    return words_register
