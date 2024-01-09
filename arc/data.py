import collections
import itertools
import logging
import math
import random
from functools import reduce
from typing import List, Union, Dict, Generator

import numpy as np
from tqdm.rich import tqdm

from arc.definitions import *
from arc.io import maybe_load_from_file
from arc.types import Word, Syllable, Register, Lexicon, Phoneme


def add_phonotactic_features(syllable_phonemes: List[Phoneme]):
    syll_feats = [[] for _ in syllable_phonemes]
    for i, phon in enumerate(syllable_phonemes):
        phon_feats = phon.info["features"]
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

    return syll_feats


def make_feature_syllables(
    phonemes: Register,
    phoneme_pattern: Union[str, list] = "cV",
    max_combinations: int = 1_000_000,
) -> Register:
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
        if phoneme.info["features"][PHONEME_FEATURE_LABELS.index('cons')] == "+":
            if len(phoneme.id) == 1:
                single_consonants.append(phoneme.id)
            else:
                multi_consonants.append(phoneme.id)
        else:
            if len(phoneme.id) == 2:
                if phoneme.info["features"][PHONEME_FEATURE_LABELS.index('long')] == "+":
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

    syllables_dict = {}
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
                if phoneme.info["features"][PHONEME_FEATURE_LABELS.index(label)] == "+":
                    syll_features.append(1)
                else:
                    syll_features.append(0)

        syllable = Syllable(
            id=syll_id, info={"binary_features": syll_features,
                              "phonotactic_features": add_phonotactic_features(syll_phons)},
            phonemes=syll_phons
        )
        syllables_dict[syll_id] = syllable

    return Register(syllables_dict)


def get_oscillation_patterns(lag):
    kernel = [1] + [0] * (lag - 1) + [1] + [0] * (lag - 1)
    return [list(np.roll(kernel, i)) for i in range(lag)]


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


def make_words(syllables, n_sylls=3, n_look_back=2, n_words=10_000, max_tries=100_000) -> Register:
    words = {}
    for _ in tqdm(range(max_tries)):
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
                word_features = list(zip(*[s.info["binary_features"] for s in sylls]))
                words[word_id] = Word(id=word_id, info={"binary_features": word_features}, syllables=sylls)

        if len(words) == n_words:
            logging.info(f"Done: Found {n_words} words.")
            break

    return Register(words)


def word_overlap_matrix(words: Register[str, Word]):
    n_words = len(words)
    n_sylls_per_word = len(words[0].syllables)

    oscillation_patterns = get_oscillation_patterns(lag=n_sylls_per_word)

    overlap = np.zeros([n_words, n_words])
    for i1, i2 in tqdm(list(itertools.product(range(n_words), range(n_words)))):
        word_pair_features = [f1 + f2 for f1, f2 in zip(words[i1].info["binary_features"],
                                                        words[i2].info["binary_features"])]

        matches = 0
        for word_pair_feature in word_pair_features:
            if word_pair_feature in oscillation_patterns:
                matches += 1

        overlap[i1, i2] = matches

    return overlap


# TODO: Make pretty from here

def sample_min_overlap_lexicon(words, overlap, n_words=6, max_overlap=1, max_yields=10):
    overlap = np.array(overlap)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    logging.info(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0

    for max_pair_overlap, max_words_with_overlap in itertools.product(range(max_overlap + 1), range(1, math.comb(n_words, 2))):

        max_cum_overlap = max_pair_overlap * max_words_with_overlap

        if max_pair_overlap != 0:
            logging.warning(f"Increasing allowed overlaps: MAX_PAIRWISE_OVERLAP={max_pair_overlap}, MAX_CUM_OVERLAP={max_cum_overlap}")

        # WORDSxWORDS boolean matrix indicating if the words can be paired together based on the maximum overlap
        valid_word_pairs_matrix = (overlap <= max_pair_overlap)

        # represent the matrix from above as a set of pairs of word indexes, e.g. {{0, 1}, {0, 2}, ...}
        valid_pairs = set(frozenset([int(pair[0]), int(pair[1])]) for pair in zip(*np.where(valid_word_pairs_matrix)))

        def check_no_syllable_overlap(pair):
            w1, w2 = pair
            intersection = set(syllable.id for syllable in words[w1]) & set(syllable.id for syllable in words[w2])
            return not intersection

        valid_pairs = set(filter(check_no_syllable_overlap, valid_pairs))

        for start_pair in valid_pairs:
            # logging.info(f"max overlap: {max_overlap}; start with pair: {i}/{len(valid_pairs)}")
            lexicon_indexes = set(start_pair)
            sum_overlaps = 0

            for candidate_idx in range(len(overlap)):
                if candidate_idx not in lexicon_indexes:
                    has_min_overlap = [({known_idx, candidate_idx} in valid_pairs) for known_idx in lexicon_indexes]

                    if all(has_min_overlap):
                        overlaps = [overlap[known_idx, candidate_idx] for known_idx in lexicon_indexes]

                        if sum(overlaps) <= (max_cum_overlap-sum_overlaps):
                            lexicon_indexes.add(candidate_idx)
                            sum_overlaps += sum(overlaps)

                            if len(lexicon_indexes) == n_words:
                                valid_sub_matrix = valid_word_pairs_matrix[np.ix_(list(lexicon_indexes), list(lexicon_indexes))]
                                assert np.all(np.int32(valid_sub_matrix) == (1 - np.eye(n_words)))
                                word_ids = []
                                word_objects = []
                                for index in lexicon_indexes:
                                    word_ids.append(words[index].id)
                                    word_objects.append(words[index])
                                lexicon = Lexicon(id="".join(word_ids), words=word_objects, info={"cumulative_overlap": sum_overlaps})

                                yield lexicon

                                yields += 1

                                if yields == max_yields:
                                    return


def make_lexicon_generator(
        words: Register[str, Word],
        n_words: int = 6,
        max_overlap: int = 1,
        max_yields: int = 10
) -> Generator[Lexicon, None, None]:
    overlap = word_overlap_matrix(words)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    logging.info(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0

    for max_pair_overlap, max_overlap_with_n_words in itertools.product(range(max_overlap + 1),
                                                                        range(1, math.comb(n_words, 2))):

        max_cum_overlap = max_pair_overlap * max_overlap_with_n_words

        if max_pair_overlap != 0:
            logging.warning(
                f"Increasing allowed overlaps: MAX_PAIRWISE_OVERLAP={max_pair_overlap}, MAX_CUM_OVERLAP={max_cum_overlap}")

        # WORDSxWORDS boolean matrix indicating if the words can be paired together based on the maximum overlap
        valid_word_pairs_matrix = (overlap <= max_pair_overlap)

        # represent the matrix from above as a set of pairs of word indexes, e.g. {{0, 1}, {0, 2}, ...}
        min_overlap_pairs = zip(*np.where(valid_word_pairs_matrix))
        min_overlap_pairs = set(frozenset([int(pair[0]), int(pair[1])]) for pair in min_overlap_pairs)

        def check_no_syllable_overlap(pair):
            w1, w2 = pair
            intersection = set(syllable.id for syllable in words[w1]) & set(syllable.id for syllable in words[w2])
            return not intersection

        min_overlap_pairs = set(filter(check_no_syllable_overlap, min_overlap_pairs))

        for start_pair in min_overlap_pairs:
            # logging.info(f"max overlap: {max_overlap}; start with pair: {i}/{len(min_overlap_pairs)}")
            lexicon_indexes = set(start_pair)
            sum_overlaps = 0

            for candidate_idx in range(len(overlap)):
                if candidate_idx not in lexicon_indexes:
                    has_min_overlap = [({known_idx, candidate_idx} in min_overlap_pairs) for known_idx in lexicon_indexes]

                    if all(has_min_overlap):
                        overlaps = [overlap[known_idx, candidate_idx] for known_idx in lexicon_indexes]

                        if sum(overlaps) <= (max_cum_overlap - sum_overlaps):
                            lexicon_indexes.add(candidate_idx)
                            sum_overlaps += sum(overlaps)

                            if len(lexicon_indexes) == n_words:
                                valid_sub_matrix = valid_word_pairs_matrix[
                                    np.ix_(list(lexicon_indexes), list(lexicon_indexes))]
                                assert np.all(np.int32(valid_sub_matrix) == (1 - np.eye(n_words)))
                                word_ids = []
                                word_objects = []
                                for index in lexicon_indexes:
                                    word_ids.append(words[index].id)
                                    word_objects.append(words[index])
                                lexicon = Lexicon(id="".join(word_ids), words=word_objects,
                                                  info={"cumulative_overlap": sum_overlaps})

                                yield lexicon

                                yields += 1

                                if yields == max_yields:
                                    return


def transitional_p_matrix(v):
    n = 1 + max(v)
    M = [[0] * n for _ in range(n)]
    for (i, j) in zip(v, v[1:]):
        M[i][j] += 1
    for r in M:
        s = sum(r)
        if s > 0:
            r[:] = [i/s for i in r]
    return M


# GENERATE SERIES OF SYLLABLES WITH UNIFORM TPs
def pseudo_walk_tp_random(P, T, v, S, N, n_words=4, n_sylls_per_word=3):
    # TODO
    t = []
    for iTrip in range(n_words):
        for iPoss in range(n_sylls_per_word):
            if not v and not t:
                if N:
                    x = [N]
                    N = []
                else:
                    x = random.sample(P[iPoss], 1)
                t += x
            else:
                iLoop = 0
                while iLoop < 1000:
                    x = random.sample([i for i in P[iPoss] if i not in t], 1)[0]
                    if not t:
                        s = v[-1]
                    else:
                        s = t[-1]
                    if [s, x] not in S and [s, x] in T[iPoss-1]:
                        S.append([s, x])
                        t.append(x)
                        break
                    else:
                        iLoop += 1
    if iLoop < 1000:
        v += t
    return v, S, t


def pseudo_rand_tp_random(n_words=4, n_sylls_per_word=3):
    # TODO
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_repetitions = n_sylls_total * 4  # number of repetitions in a trial
    P = [list(range(i, n_sylls_total, n_sylls_per_word)) for i in range(n_sylls_per_word)]
    I = list(range(n_sylls_per_word))
    B = I[:]
    B.append(B.pop(0))
    T = [[list(i) for i in list(itertools.product(*[P[j], P[k]]))] for j, k in zip(I, B)]
    while True:
        V = []
        while len(V) < n_sylls_total * n_repetitions:
            v = []
            t = []
            if not V:
                N = []
                S = []
            else:
                N = [i for i in T[2] if i not in S][0][1]
                S = []
            while len(v) < len(T) * len(T[0]):
                v, S, t = pseudo_walk_tp_random(P, T, v, S, N)
                if len(t) < n_sylls_total:
                    v = []
                    S = []
                    t = []
            V += v
        V.append(V[0])
        M = np.array(transitional_p_matrix(V))
        V.pop()
        if all(set(i) <= set([1/n_words, 0]) for i in M):
            break
    return V, M


# GENERATE RANDOMIZATIONS CONTROLLING FOR UNIFORMITY OF TRANSITION PROBABILITIES ACROSS WORDS
def pseudo_rand_tp_struct(n_words=4, n_sylls_per_word=3):
    # TODO
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_repetitions = n_sylls_total * 4  # number of repetitions in a trial
    while True:
        P = list(itertools.permutations(np.arange(n_words)))
        S = [[*(i for _ in range(math.ceil(n_repetitions / len(P))))] for i in P]
        M = np.zeros((n_words, n_words))
        v = []
        c = 2
        mask = (np.ones((n_words, n_words)) - np.diag(np.ones(n_words))).astype(bool)
        for _ in range(n_repetitions):
            R = [S[i] for i in range(len(S)) if len(S[i]) == max(map(len, S))]
            if not v:
                p = list(list(random.sample(R, 1)[0])[0])
                m = np.array(transitional_p_matrix(p))
            else:
                s = v[-1]
                n = list(np.where(M[s] == M[s].min())[0])
                n.remove(s)
                if not n:
                    n = list(np.where(M[s] == np.sort(M[s])[1])[0])
                T = [j for j in P if any(k for k in [(s, i) for i in n]) in bytes(j) and j[0] != s]
                t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                if not t:
                    t = T
                while True:
                    if collections.Counter(M[mask])[c - 1] >= len(M[mask]) - n_words:
                        c += 1
                    if t:
                        p = list(random.sample(t, 1)[0])
                        m = np.array(transitional_p_matrix(p))
                        m[s][p[0]] = m[s][p[0]] + 1
                        if c not in M + m:
                            break
                        else:
                            t.remove(tuple(p))
                    else:
                        c += 1
                        T = [i for i in P if any(j for j in [(s, k) for k in n]) in bytes(i) and i[0] != s]
                        t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                        if not t:
                            t = T
            M += m
            v += p
            r = P.index(tuple(p))
            if tuple(p) in S[r]:
                S[r].remove(tuple(p))
            else:
                break
        if all(np.diagonal(M) == np.zeros(n_words)) and len(v) == n_words * n_repetitions:
            if M[mask].min() == math.floor(n_repetitions * n_words / len(M[mask])) - 1:
                if M[mask].max() == math.ceil(n_repetitions * n_words / len(M[mask])):
                    v.append(v[0])
                    M = np.array(transitional_p_matrix(v))
                    v.pop()
                    break
    return v, M


@maybe_load_from_file(path=os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle'), force_redo=False)
def generate_stream_randomization():
    logging.info("GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs")
    TP_struct_V = []
    for _ in tqdm(range(N_RANDOMIZATIONS_PER_STREAM)):
        while True:
            v_struct, m_struct = pseudo_rand_tp_struct()
            if v_struct not in TP_struct_V:
                TP_struct_V.append(v_struct)
                break

    TP_random_V = []
    for _ in tqdm(range(N_RANDOMIZATIONS_PER_STREAM)):
        while True:
            v_random, m_struct = pseudo_rand_tp_random()
            if v_random not in TP_random_V:
                TP_random_V.append(v_random)
                break

    return TP_struct_V, TP_random_V
