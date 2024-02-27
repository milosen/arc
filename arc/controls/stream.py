import collections
import itertools
import logging
import math
import random
from copy import copy
from typing import List, Optional, Literal, Tuple, Dict

import numpy as np

from arc.controls.common import get_oscillation_patterns
from arc.controls.lexicon import make_lexicon_generator
from arc.core.base_types import Register
from arc.core.syllable import Syllable
from arc.core.word import WordType, Word

StreamType = WordType
Stream = Word


def transitional_p_matrix(v):
    # TODO: make (faster and) easier to read
    n = 1 + max(v)
    M = [[0] * n for _ in range(n)]
    for (i, j) in zip(v, v[1:]):
        M[i][j] += 1
    for r in M:
        s = sum(r)
        if s > 0:
            r[:] = [i/s for i in r]
    return M


def pseudo_walk_tp_random(P, T, v, S, N, n_words=4, n_sylls_per_word=3):
    # TODO: make faster and easier to read
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
    # TODO: make faster and easier to read
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


def pseudo_rand_tp_struct(n_words=4, n_sylls_per_word=3):
    # TODO: make faster and easier to read
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


def sample_syllable_randomization(
        lexicon: Register[str, Word],
        max_tries=1000,
        tp_mode: Literal["word_structured", "position_controlled", "random"] = "word_structured") -> List[Syllable]:
    if tp_mode == "word_structured":
        elements = [word for word in lexicon]
        rand_func = pseudo_rand_tp_struct
    elif tp_mode == "position_controlled":
        elements = [syllable for word in lexicon for syllable in word]
        rand_func = pseudo_rand_tp_random
    else:
        raise ValueError(f"tp_mode '{tp_mode}' unknown.")

    randomized_indexes_list = []
    n_syllables = len(lexicon[0].syllables)
    n_words = len(lexicon)

    for _ in range(max_tries):
        randomized_indexes, _ = rand_func(n_words=n_words, n_sylls_per_word=n_syllables)
        if randomized_indexes not in randomized_indexes_list:
            randomized_indexes_list.append(randomized_indexes)
            if tp_mode == "word_structured":
                yield [syllable for index in randomized_indexes for syllable in elements[index]]
            else:
                yield [elements[index] for index in randomized_indexes]


def compute_rhythmicity_index_sylls_stream(stream, patterns):
    count_patterns = []
    patterns = [tuple(pat) for pat in patterns]
    for feature_stream in list(zip(*[syllable.info["binary_features"] for syllable in stream])):
        c = 0
        for iSyll in range(len(feature_stream) - max(len(i) for i in patterns)):
            if any(i == feature_stream[iSyll: iSyll + len(i)] for i in patterns):
                c += 1
        count_patterns.append(c / (len(feature_stream) - max(len(i) for i in patterns)))

    return count_patterns


def make_stream_from_lexicon(lexicon: Register[str, Word],
                             max_rhythmicity=0.1, max_tries_randomize=10,
                             tp_mode: Literal["word_structured", "position_controlled", "random"] = "word_structured"):

    for sylls_stream in sample_syllable_randomization(lexicon, max_tries=max_tries_randomize, tp_mode=tp_mode):
        patterns = get_oscillation_patterns(len(lexicon[0].syllables))
        rhythmicity_indexes = compute_rhythmicity_index_sylls_stream(sylls_stream, patterns)

        if max(rhythmicity_indexes) <= max_rhythmicity:
            i_labels = enumerate(lexicon.info["syllable_feature_labels"])
            feature_labels = [f"phon_{i_phon+1}_{label}" for i_phon, labels in i_labels for label in labels]

            return Stream(
                id="".join([syll.id for syll in sylls_stream]),
                syllables=sylls_stream,
                info={
                    "rhythmicity_indexes": {k: float(v) for k, v in zip(feature_labels, rhythmicity_indexes)},
                    "lexicon": lexicon,
                    "stream_tp_mode": tp_mode,
                    **lexicon.info,
                }
            )


def make_stream_from_words(words: Register[str, Word],
                           n_words: int = 4,
                           max_word_overlap: int = 1,
                           max_lexicons: int = 10,
                           max_rhythmicity=0.1,
                           max_tries_randomize=10,
                           tp_mode: Literal["word_structured", "position_controlled", "random"] = "word_structured"
                           ) -> Optional[StreamType]:

    l_gen = make_lexicon_generator(words=words,
                                   n_words=n_words, max_overlap=max_word_overlap, max_yields=max_lexicons)

    for lex in l_gen:
        maybe_stream: Optional[StreamType] = make_stream_from_lexicon(lex, max_rhythmicity=max_rhythmicity,
                                                                      max_tries_randomize=max_tries_randomize,
                                                                      tp_mode=tp_mode)

        if maybe_stream:
            return maybe_stream


def make_compatible_streams(words: Register[str, Word],
                            n_words: int = 4,
                            max_word_overlap: int = 1,
                            max_lexicons: int = 10,
                            max_rhythmicity=0.1,
                            max_tries_randomize=10) -> Tuple:
    logging.info("Building streams from a pair of compatible lexicons...")

    lexicon_generator_1 = make_lexicon_generator(
        words=words, n_words=n_words, max_overlap=max_word_overlap, max_yields=max_lexicons)

    lexicon_generator_2 = make_lexicon_generator(
        words=words, n_words=n_words, max_overlap=max_word_overlap, max_yields=max_lexicons)

    # pairwise lexicon generation
    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        if not set(lexicon_1.keys()).intersection(set(lexicon_2.keys())):
            logging.info("Dropping Lexicons because they have overlapping syllables.")
            continue

        maybe_stream_1_words: Optional[StreamType] = make_stream_from_lexicon(lexicon_1,
                                                                              max_rhythmicity=max_rhythmicity,
                                                                              max_tries_randomize=max_tries_randomize,
                                                                              tp_mode="word_structured")

        if not maybe_stream_1_words:
            logging.info("Dropping Lexicons because no good word-randomized stream for Lexicon 1 was found.")
            continue

        maybe_stream_1_sylls: Optional[StreamType] = make_stream_from_lexicon(lexicon_1,
                                                                              max_rhythmicity=max_rhythmicity,
                                                                              max_tries_randomize=max_tries_randomize,
                                                                              tp_mode="position_controlled")

        if not maybe_stream_1_sylls:
            logging.info("Dropping Lexicons because no good syllable-randomized stream for Lexicon 1 was found.")
            continue

        maybe_stream_2_words: Optional[StreamType] = make_stream_from_lexicon(lexicon_1,
                                                                              max_rhythmicity=max_rhythmicity,
                                                                              max_tries_randomize=max_tries_randomize,
                                                                              tp_mode="word_structured")

        if not maybe_stream_2_words:
            logging.info("Dropping Lexicons because no good syllable-randomized stream for Lexicon 1 was found.")
            continue

        maybe_stream_2_sylls: Optional[StreamType] = make_stream_from_lexicon(lexicon_1,
                                                                              max_rhythmicity=max_rhythmicity,
                                                                              max_tries_randomize=max_tries_randomize,
                                                                              tp_mode="position_controlled")

        if not maybe_stream_2_sylls:
            logging.info("Dropping Lexicons because no good syllable-randomized stream for Lexicon 2 was found.")
            continue

        return maybe_stream_1_words, maybe_stream_1_sylls, maybe_stream_2_words, maybe_stream_2_sylls

    return tuple()


def get_stream_syllable_stats(stream: StreamType) -> Dict:
    d = {}
    for syllable in stream:
        if syllable.id in d:
            d[syllable.id] += 1
        else:
            d[syllable.id] = 1

    return d
