import itertools
import logging
import random
from functools import reduce
from typing import List, Union, Dict

from tqdm.rich import tqdm

from arc.definitions import *
from arc.io import read_phoneme_features
from arc.types import Word, Syllable, CollectionARC


def check_syll_feature_overlap(syllables):
    all_feats = [feat for syll in syllables for phon_feats in syll.phonotactic_features for feat in phon_feats]
    return len(all_feats) == len(set(all_feats))


def generate_subset_syllables(syllables, lookback_syllables):
    subset = []
    for new_syll in syllables:
        syll_test_set = lookback_syllables + [new_syll]
        if check_syll_feature_overlap(syll_test_set):
            subset.append(new_syll)

    return subset


def make_words(syllables, n_sylls=3, n_look_back=2, max_tries=10_000) -> CollectionARC:
    words = {}
    for _ in tqdm(range(max_tries)):
        sylls = []
        for _ in range(n_sylls):
            sub = generate_subset_syllables(syllables, sylls[-n_look_back:])
            if sub:
                syll = random.sample(sub, 1)[0]
                sylls.append(syll)

        if len(sylls) == n_sylls:
            word_id = "".join(s.id for s in sylls)
            word_features = list(zip(*[s.binary_features for s in sylls]))

            if word_id not in words:
                words[word_id] = Word(id=word_id, info={}, syllables=sylls, binary_features=word_features)

    return CollectionARC(words)


def add_phonotactic_features(syllable: Syllable):
    syll_feats = [[] for _ in syllable.phonemes]
    for i, phon in enumerate(syllable.phonemes):
        phon_feats = phon.features
        is_consonant = phon_feats[PHONEME_FEATURE_LABELS.index("cons")] == "+"
        if is_consonant:
            if phon_feats[SON] == '+':
                syll_feats[i].append("son")
            if phon_feats[SON] != '+' and phon_feats[CONT] != '+':
                syll_feats[i].append("plo")
            if phon_feats[SON] != '+' and phon_feats[CONT] == '+':
                syll_feats[i].append("fri")
            if phon_feats[LAB] == '+':
                syll_feats[i].append("lab")
            if phon_feats[COR] == '+' and phon_feats[HI] != '+':
                syll_feats[i].append("den")
            if "lab" not in syll_feats[i] and "den" not in syll_feats[i]:
                syll_feats[i].append("oth")
        else:
            for vowel in ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ø', 'y']:
                if vowel in phon.id:
                    syll_feats[i].append(vowel)

    syllable.phonotactic_features = syll_feats

    return syllable


def make_feature_syllables(
    phonemes: CollectionARC,
    phoneme_pattern: Union[str, list] = "cV",
    max_combinations: int = 1_000_000,
) -> CollectionARC:
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
        if phoneme.features[PHONEME_FEATURE_LABELS.index('cons')] == "+":
            if len(phoneme.id) == 1:
                single_consonants.append(phoneme.id)
            else:
                multi_consonants.append(phoneme.id)
        else:
            if len(phoneme.id) == 2:
                if phoneme.features[PHONEME_FEATURE_LABELS.index('long')] == "+":
                    long_vowels.append(phoneme.id)
                else:
                    short_vowels.append(phoneme.id)

    phonemes_mapping = {"c": single_consonants, "C": multi_consonants, "v": short_vowels, "V": long_vowels}

    phonemes_factors = list(map(lambda phoneme_type: phonemes_mapping[phoneme_type], phoneme_types))
    total_combs = reduce(lambda a, b: a * b, [len(phonemes_factor) for phonemes_factor in phonemes_factors])
    if total_combs > max_combinations:
        logging.warning(f"Combinatorial explosion with {total_combs} combinations for '{phoneme_types}'."
                        f"I will only generate {max_combinations} of them, but you can set this number higher with the "
                        "option 'max_combinations'.")

    syllables_phoneme_comb = {}
    list_of_combinations = []
    for i, phoneme_combination in enumerate(itertools.product(*phonemes_factors)):
        if i >= max_combinations:
            break
        list_of_combinations.append(phoneme_combination)

    for phoneme_combination in tqdm(list_of_combinations):
        syll_id = "".join(phoneme_combination)
        syll_phons = []
        syll_features = []
        for p, phoneme_feature_labels in zip(phoneme_combination, syll_feature_labels):
            phoneme = phonemes[p]
            syll_phons.append(phoneme)
            for label in phoneme_feature_labels:
                if phoneme.features[PHONEME_FEATURE_LABELS.index(label)] == "+":
                    syll_features.append(1)
                else:
                    syll_features.append(0)

        syllable = Syllable(
            id=syll_id, info={}, phonemes=syll_phons, binary_features=syll_features, phonotactic_features=[]
        )
        syllables_phoneme_comb[syll_id] = add_phonotactic_features(syllable)

    return CollectionARC(syllables_phoneme_comb)
