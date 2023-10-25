import collections
import itertools
import math
import random
from typing import Callable, Iterable

import numpy as np

from arc.definitions import LABELS_C, LABELS_V, N_FEAT


def filter_iterable(function: Callable, iterable: Iterable):
    return list(filter(function, iterable))


def map_iterable(function: Callable, iterable: Iterable):
    return list(map(function, iterable))


def transitional_p_matrix(V):
    n = 1 + max(V)
    M = [[0] * n for _ in range(n)]
    for (i, j) in zip(V, V[1:]):
        M[i][j] += 1
    for r in M:
        s = sum(r)
        if s > 0:
            r[:] = [i/s for i in r]
    return M


# GENERATE SERIES OF SYLLABLES WITH UNIFORM TPs
def pseudo_walk_TP_random(P, T, v, S, N, n_words=4, n_sylls_per_word=3):
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


def pseudo_rand_TP_random(n_words=4, n_sylls_per_word=3):
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
                v, S, t = pseudo_walk_TP_random(P, T, v, S, N)
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
def pseudo_rand_TP_struct(n_words=4, n_sylls_per_word=3):
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


# COMPUTE EMPIRICAL RHYTHMICITY INDEX AT THE FREQUENCY OF INTEREST FOR A SEQUENCE OF SYLLABLES
def compute_rhythmicity_index(trial, patts, numbs, phons, labels):
    cnsnt = [i[0] for i in trial]
    vowel = [i[1:] for i in trial]
    stationarity_index = []
    for iChar in [cnsnt, vowel]:
        feats = [numbs[phons.index(i)] for i in iChar]
        count_patterns = []
        if iChar == cnsnt:
            nFeat = [i for i in range(len(labels)) if labels[i] in LABELS_C]
        else:
            nFeat = [i for i in range(len(labels)) if labels[i] in LABELS_V]
        for iFeat in nFeat:
            v = []
            for iSyll in range(len(trial)):
                if feats[iSyll][iFeat] == '+':
                    v.append(1)
                else:
                    v.append(0)
            v = tuple(v)
            c = 0
            for iSyll in range(len(v) - max(len(i) for i in patts)):
                if any(i == v[iSyll : iSyll + len(i)] for i in patts):
                    c += 1
            count_patterns.append(c / (len(v) - max(len(i) for i in patts)))
        stationarity_index.append(count_patterns)
    stationarity_index = list(itertools.chain.from_iterable(stationarity_index))
    return stationarity_index


# COMPUTE FEATURES SIMILARITY AT THE FREQUENCY OF INTEREST FOR A GIVEN PAIR OF WORDS
def compute_features_overlap(word_pair_features, patterns, c_sum: bool = True):
    match = []
    for word_pair_feature in word_pair_features:
        if word_pair_feature in patterns:
            match.append(1)
        else:
            match.append(0)
    if c_sum:
        return sum(match)
    else:
        return match


# COMPUTE FEATURES OVERLAP FOR EACH PAIR OF WORDS
def compute_word_overlap_matrix(words, features, oscillation_patterns):
    overlap = []
    for idx_1 in range(len(words)):
        ovlap = []
        for idx_2 in range(len(words)):
            word_pair_features = [tuple(i + j) for i, j in zip(features[idx_1], features[idx_2])]
            ovlap.append(compute_features_overlap(word_pair_features, oscillation_patterns))
        overlap.append(ovlap)
    return overlap


# COMPUTE THEORETICAL RHYTHMICITY INDEX AT THE FREQUENCY OF INTEREST FOR A SET OF WORDS
def theoretical_feats_idx(set_i, feats, patts, n_words=4, n_sylls_per_word=3):
    ov_theory = []
    for idx_A in range(n_words):
        for idx_B in range(n_words):
            iComb = [tuple(i + j) for i, j in
                      zip(feats[idx_A], feats[idx_B])]
            if idx_A != idx_B:
                ovlap = compute_features_overlap(iComb, patts, False)
                ovlap = [i for i in range(N_FEAT) if ovlap[i] == 1]
                ov_theory.append(ovlap)
    ov_theory = [i for i in ov_theory if i]
    ov_theory = list(itertools.chain.from_iterable(ov_theory))
    ov_theory = collections.Counter(ov_theory)
    ov_theory = [int(ov_theory[i] / 2) for i in range(N_FEAT)]
    if sum(ov_theory) <= n_sylls_per_word * 2 and not any(i > 1 for i in ov_theory):
        ri_theory = []
        perms = list(itertools.permutations(feats))
        for iPerm in perms:
            iComb = [tuple(i + j + k + m) for i, j, k, m in
                     zip(iPerm[0], iPerm[1], iPerm[2], iPerm[3])]
            count = []
            for iFeat in range(N_FEAT):
                c = 0
                for iSyll in range(len(iComb[iFeat]) - n_sylls_per_word * 2):
                    part_v = iComb[iFeat][iSyll : iSyll + n_sylls_per_word * 2]
                    if any(i_Pat == part_v for i_Pat in patts):
                        c += 1
                count.append(c / (len(iComb[iFeat]) - n_sylls_per_word * 2))
            ri_theory.append(count)
        ri_theory = np.array(ri_theory)
        ri_theory = np.mean(ri_theory, axis = 0).tolist()
        if all(i < 0.1 for i in ri_theory):
            good_idxs = set_i
        else:
            good_idxs = False; ov_theory = False; ri_theory = False
    else:
        good_idxs = False; ov_theory = False; ri_theory = False
    return good_idxs, ov_theory, ri_theory


# EXTRACT BINARY FEATURE MATRIX FOR EACH PHONEME IN A SEQUENCE OF CV SYLLABLES
def binary_feature_matrix(word, numbs, phons, labels):
    sylls = [s.syll for s in word]
    cnsnt = [i[0] for i in sylls]
    vowel = [i[1:] for i in sylls]
    Feats = []
    for iChar in [cnsnt, vowel]:
        fChar = []
        feats = [numbs[phons.index(i)] for i in iChar]
        if iChar == cnsnt:
            nFeat = [i for i in range(len(labels)) if labels[i] in LABELS_C]
        else:
            nFeat = [i for i in range(len(labels)) if labels[i] in LABELS_V]
        for iFeat in nFeat:
            v = []
            for iSyll in range(len(sylls)):
                if feats[iSyll][iFeat] == '+':
                    v.append(1)
                else:
                    v.append(0)
            fChar.append(v)
        Feats.append(fChar)
    Feats = list(itertools.chain.from_iterable(Feats))
    return Feats
