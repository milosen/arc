import itertools
import logging
from copy import copy
from functools import reduce
from typing import List, Union

from tqdm.rich import tqdm

from arc.types import Syllable, Register, Phoneme, TypeRegister


LABELS_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
LABELS_V = ['back', 'hi', 'lo', 'lab', 'tense', 'long']
N_FEAT = len(LABELS_C) + len(LABELS_V)  # 14


def add_phonotactic_features(syllable_phonemes: List[Phoneme]):
    syll_feats = [[] for _ in syllable_phonemes]
    for i, phon in enumerate(syllable_phonemes):
        is_consonant = phon.get_binary_feature("cons")
        if is_consonant:
            if phon.get_binary_feature("son"):
                syll_feats[i].append("son")
            if not phon.get_binary_feature("son") and not phon.get_binary_feature("cont"):
                syll_feats[i].append("plo")
            if not phon.get_binary_feature("son") and phon.get_binary_feature("cont"):
                syll_feats[i].append("fri")
            if phon.get_binary_feature("lab"):
                syll_feats[i].append("lab")
            if phon.get_binary_feature("cor") and not phon.get_binary_feature("hi"):
                syll_feats[i].append("den")
            if "lab" not in syll_feats[i] and "den" not in syll_feats[i]:
                syll_feats[i].append("oth")
        else:
            for vowel in ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ø', 'y']:
                if vowel in phon.id:
                    syll_feats[i].append(vowel)

    return syll_feats


def make_feature_syllables(
    phonemes: TypeRegister,
    phoneme_pattern: Union[str, list] = "cV",
    max_combinations: int = 1_000_000,
    progress_bar: bool = False,
) -> TypeRegister:
    """Generate syllables form feature-phonemes. Only keep syllables that follow the phoneme pattern"""

    logging.info("SELECT SYLLABLES WITH GIVEN PHONEME-TYPE PATTERN AND WITH PHONEMES WE HAVE FEATURES FOR")
    valid_phoneme_types = ["c", "C", "v", "V"]

    phoneme_types_user = list(phoneme_pattern) if isinstance(phoneme_pattern, str) else phoneme_pattern

    phoneme_types = list(filter(lambda p: p in valid_phoneme_types, phoneme_types_user))

    if phoneme_types_user != phoneme_types:
        logging.warning(f"ignoring invalid phoneme types {phoneme_types_user} -> {phoneme_types}. "
              f"You can use the following phoneme types in your pattern: {valid_phoneme_types}")

    logging.info(f"Search for phoneme-pattern '{''.join(phoneme_types)}'")

    labels_mapping = {"c": LABELS_C, "C": LABELS_C, "v": LABELS_V, "V": LABELS_V}
    syll_feature_labels = list(map(lambda t: labels_mapping[t], phoneme_types))

    single_consonants, multi_consonants, short_vowels, long_vowels = [], [], [], []
    for phoneme in phonemes.values():
        if phoneme.get_binary_feature('cons'):
            if len(phoneme.id) == 1:
                single_consonants.append(phoneme.id)
            else:
                multi_consonants.append(phoneme.id)
        else:
            if len(phoneme.id) == 2:
                if phoneme.get_binary_feature('long'):
                    long_vowels.append(phoneme.id)
                else:
                    short_vowels.append(phoneme.id)

    phonemes_mapping = {"c": single_consonants, "C": multi_consonants, "v": short_vowels, "V": long_vowels}

    phonemes_factors = list(map(lambda phoneme_type: phonemes_mapping[phoneme_type], phoneme_types))
    total_combs = reduce(lambda a, b: a * b, [len(phonemes_factor) for phonemes_factor in phonemes_factors])
    if total_combs > max_combinations:
        logging.warning(f"Combinatorial explosion with {total_combs} combinations for '{phoneme_types}'."
                        f"I will only generate {max_combinations} of them, but you can set this number higher via the "
                        "option 'max_combinations'.")

    syllables_dict = {}
    list_of_combinations = []
    for i, phoneme_combination in enumerate(itertools.product(*phonemes_factors)):
        if i >= max_combinations:
            break
        list_of_combinations.append(phoneme_combination)

    if progress_bar:
        list_of_combinations = tqdm(list_of_combinations)

    for phoneme_combination in list_of_combinations:
        syll_id = "".join(phoneme_combination)
        syll_phons = []
        syll_features = []
        for p, phoneme_feature_labels in zip(phoneme_combination, syll_feature_labels):
            phoneme: Phoneme = phonemes[p]
            syll_phons.append(phoneme)
            for label in phoneme_feature_labels:
                if phoneme.get_binary_feature(label):
                    syll_features.append(1)
                else:
                    syll_features.append(0)

        syllable = Syllable(
            id=syll_id, info={"binary_features": syll_features,
                              "phonotactic_features": add_phonotactic_features(syll_phons)},
            phonemes=syll_phons
        )
        syllables_dict[syll_id] = syllable

    new_info = copy(phonemes.info)
    new_info.update({"syllable_feature_labels": syll_feature_labels,  "phoneme_pattern": phoneme_pattern})

    return Register(syllables_dict, _info=new_info)
