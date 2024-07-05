import csv
import json
import logging
import os
import pathlib
from importlib import resources as importlib_resources
from os import PathLike
from typing import Iterable, Dict, Union, List, Type, Optional, Literal
from functools import partial
from copy import copy

import numpy as np

from arc.io import load_phonemes
from arc.types.base_types import Register, RegisterType

from arc.types.phoneme import PHONEME_FEATURE_LABELS, Phoneme
from arc.types.syllable import Syllable
from arc.types.word import Word
from arc.types.lexicon import LexiconType
from arc.types.stream import Stream

from arc.core.syllable import LABELS_C, LABELS_V, syllable_from_phonemes
from arc.core.word import Word, word_overlap_matrix
from arc.core.stream import compute_rhythmicity_index_sylls_stream, get_oscillation_patterns

DEFAULT_PHONEMES = load_phonemes()
SYLLABLE_FEAT_LABELS = [LABELS_C] + [LABELS_V]

def to_syllable(syllable, syllable_type="cv"):
    says_cv = (syllable_type == "cv")
    says_cV = (syllable_type == "cV")

    is_cV = (len(syllable) == 3) and syllable.endswith("ː")
    is_cv = (len(syllable) == 2) and not syllable.endswith("ː")
    is_diphthong = (len(syllable) == 3 and not syllable.endswith("ː"))

    if not any([says_cv and is_cv, says_cV and is_cV, says_cv and is_diphthong]):
        raise ValueError(f"The syllable type '{syllable_type}' does not match for syllable {syllable}. "
                          "The types 'cv' (single character consonant + short vowel) and 'cV' (single "
                          "character consonant + long vowel) are supported. Short vowel can also be a diphthong. " 
                          "All syllables must be of that type.")
    
    if is_diphthong:
        syllable_obj = syllable_from_phonemes(DEFAULT_PHONEMES, syllable[:2], SYLLABLE_FEAT_LABELS)
        syllable_obj.id = syllable
        return syllable_obj
        
    return syllable_from_phonemes(DEFAULT_PHONEMES, syllable, SYLLABLE_FEAT_LABELS)

def to_word(word, syllable_type="cv"):
    to_syllable_partial = partial(to_syllable, syllable_type=syllable_type)
    syllables_list = list(map(to_syllable_partial, word))
    word_id = "".join(s.id for s in syllables_list)
    word_features = list(list(tup) for tup in zip(*[s.info["binary_features"] for s in syllables_list]))
    return Word(id=word_id, info={"binary_features": word_features}, syllables=syllables_list)

def to_lexicon(lexicon, syllable_type="cv"):
    to_word_partial = partial(to_word, syllable_type=syllable_type)
    word_objs_list = list(map(to_word_partial, lexicon))
    lexicon = Register({w.id:  w for w in word_objs_list})
    lexicon.info.update({"syllable_feature_labels": [LABELS_C, LABELS_V],  "syllable_type": syllable_type})
    overlap = word_overlap_matrix(lexicon)
    lexicon.info["cumulative_feature_repetitiveness"] = np.triu(overlap, 1).sum()
    lexicon.info["max_pairwise_feature_repetitiveness"] = np.triu(overlap, 1).max()
    return lexicon

def to_stream(stream, syllable_type="cv", lag_of_interest=3):
    if syllable_type not in ["cv", "cV"]:
        raise ValueError(f"Syllable type {syllable_type} not supported, only one of ['cv', 'cV'].")
    to_syllable_partial = partial(to_syllable, syllable_type=syllable_type)
    syllables_list = list(map(to_syllable_partial, stream))
    stream_id = "".join(s.id for s in syllables_list)

    stream = Stream(id=stream_id, info={"lexicon": None}, syllables=syllables_list)

    patterns = get_oscillation_patterns(lag_of_interest)
    rhythmicity_indexes = compute_rhythmicity_index_sylls_stream(stream, patterns)
    i_labels = enumerate(SYLLABLE_FEAT_LABELS)
    feature_labels = [f"phon_{i_phon+1}_{label}" for i_phon, labels in i_labels for label in labels]
    stream.info.update({"rhythmicity_indexes": {k: float(v) for k, v in zip(feature_labels, rhythmicity_indexes)}})

    return stream
