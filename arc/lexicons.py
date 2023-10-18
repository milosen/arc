import csv
import os
import pickle

import numpy as np
from scipy.stats import stats
from tqdm.rich import tqdm

from arc import RESULTS_DEFAULT_PATH
from arc.phonecodes import phonecodes
from arc.syllables import read_binary_features, BINARY_FEATURES_DEFAULT_PATH


def sample_lexicon_with_minimal_overlap(words, sylls, feats, ovlap):
    pass


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:42:25 2023

This script generates artificial lexicons composed of phonologically dissimilar pseudo-words:
(1)  CV syllables with homogenous frequency of occurrence in a German corpus are sub-selected
(2)  Phonetic n-gram statistics are extracted from a German corpus (Arnold & Tomaschek, 2016)
(2)  Phonological features of all CV syllables are extracted from a matrix of German phonemes
(3)  Words are generated obeying phonotactics constraints: Obligatory Contour Principle (OCP)
(4)  Words with homogenous bigram and trigram phoneme frequency of co-occurrence are selceted
(5)  Artificial words that cannot be mistaken for real words are selected by a native speaker
(6)  A phonemic similarity matrix is computed bv measuring features overlap of all word pairs
(7)  Lexicons are computed as sets of words with minimum rhythmicity of phonological features
(8)  Streams with rhythmic or homogenous patterns of transitional probabilities are generated
(9)  The rhythmicity index of phonological features at the word-rate is calculated per stream
(10) Streams with the minimum rhythmicity index across all phonological features are selected

@author: titone
"""

# PROJECT DIRECTORY
project_dir = '/'

#%%#################################################################################################
############################################## IMPORT ##############################################
####################################################################################################

# IMPORT MODULES
import csv
import math
import pickle
import random
# import phonecodes
import numpy as np
import itertools as it
from scipy import stats
from collections import Counter as cnt

#%%#################################################################################################
########################################## SET PARAMETERS ##########################################
####################################################################################################

# GLOBAL PARAMETERS TO GENERATE TRIPLETS OF SYLLABLES
nTrip = 4               # number of words in every lexicon
nPoss = 3               # number of syllables in a triplet
nChar = 3               # number of characters in syllable
nPhon = 2               # number of phonemes in a syllable
nTrls = 1               # number of streams per conditions
nSets = 1               # number of lexicon sets to obtain
nSyll = nPoss * nTrip   # number of syllables in a lexicon
nReps = nSyll * 4       # number of repetitions in a trial
nRand = 100             # number of randomizations streams ('pseudo_rand.pickle' has 100000)

#%%#################################################################################################
######################################### CUSTOM FUNCTIONS #########################################
####################################################################################################

# !!! GENERATE A SUBSET OF SYLLABLES WITH NO OVERLAP OF CLASSES OF FEATURES WITH PREVIOUS SYLLABLES
def generate_subset_sylls(iSyll, allIdx, f_Cls):
    if iSyll in f_Cls[0][0]:
        set_1 = allIdx - f_Cls[0][0]
    elif iSyll in f_Cls[0][1]:
        set_1 = allIdx - f_Cls[0][1]
    elif iSyll in f_Cls[0][2]:
        set_1 = allIdx - f_Cls[0][2]
    if iSyll in f_Cls[1][0]:
        set_2 = allIdx - f_Cls[1][0]
    elif iSyll in f_Cls[1][1]:
        set_2 = allIdx - f_Cls[1][1]
    elif iSyll in f_Cls[1][2]:
        set_2 = allIdx - f_Cls[1][2]
    if iSyll in f_Cls[2][0]:
        set_3 = allIdx - f_Cls[2][0]
    elif iSyll in f_Cls[2][1]:
        set_3 = allIdx - f_Cls[2][1]
    elif iSyll in f_Cls[2][2]:
        set_3 = allIdx - f_Cls[2][2]
    elif iSyll in f_Cls[2][3]:
        set_3 = allIdx - f_Cls[2][3]
    elif iSyll in f_Cls[2][4]:
        set_3 = allIdx - f_Cls[2][4]
    elif iSyll in f_Cls[2][5]:
        set_3 = allIdx - f_Cls[2][5]
    elif iSyll in f_Cls[2][6]:
        set_3 = allIdx - f_Cls[2][6]
    elif iSyll in f_Cls[2][7]:
        set_3 = allIdx - f_Cls[2][7]
    set_i = set_1.intersection(set_1, set_2, set_3)
    return set_i

# !!! GENERATE TRISYLLABIC WORDS WITH NO OVERLAP OF CLASSES OF FEATURES ACROSS SYLLABLES
def generate_trisyll_word(Sylls):
    allIdx = set([CVsyl.index(i) for i in Sylls])
    while True:
        syl_1 = CVsyl.index(random.sample(Sylls, 1)[0])
        set_1 = generate_subset_sylls(syl_1, allIdx, f_Cls)
        if set_1:
            syl_2 = random.sample(list(set_1), 1)[0]
            set_2 = generate_subset_sylls(syl_2, allIdx, f_Cls)
            if set_2:
                set_3 = set_1.intersection(set_1, set_2)
                if set_3:
                    syl_3 = random.sample(list(set_3), 1)[0]
                    break
    iWord = CVsyl[syl_1] + CVsyl[syl_2] + CVsyl[syl_3]
    return iWord

# CHECK IF A TRIPLET HAS NON-ZERO (gram) OR UNIFORM (Gram) BIGRAM AND TRIGRAM LOG-PROBABILITY
def get_corpus_gram_stats(iTrip, Gram2, Gram3):
    GoodT = True
    seg_1 = iTrip[1:4]
    seg_2 = iTrip[4:7]
    if any(i not in Gram2 for i in [seg_1, seg_2]):
        GoodT = False
    seg_1 = iTrip[0:4]
    seg_2 = iTrip[1:6]
    seg_3 = iTrip[3:7]
    seg_4 = iTrip[4:9]
    if any(i not in Gram3 for i in [seg_1, seg_2, seg_3, seg_4]):
        GoodT = False
    return GoodT

# EXTRACT BINARY FEATURE MATRIX FOR EACH PHONEME IN A SEQUENCE OF CV SYLLABLES
def binary_feature_matrix(sylls, numbs, phons, labls):
    lbl_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
    lbl_V = ['back', 'hi', 'lo', 'lab', 'tense']
    cnsnt = [i[0] for i in sylls]
    vowel = [i[1:] for i in sylls]
    Feats = []
    for iChar in [cnsnt, vowel]:
        fChar = []
        feats = [numbs[phons.index(i)] for i in iChar]
        if iChar == cnsnt:
            nFeat = [i for i in range(len(labls)) if labls[i] in lbl_C]
        else:
            nFeat = [i for i in range(len(labls)) if labls[i] in lbl_V]
        for iFeat in nFeat:
            v = []
            for iSyll in range(len(sylls)):
                if feats[iSyll][iFeat] == '+':
                    v.append(1)
                else:
                    v.append(0)
            fChar.append(v)
        Feats.append(fChar)
    Feats = list(it.chain.from_iterable(Feats))
    return Feats

# COMPUTE FEATURES SIMILARITY AT THE FREQUENCY OF INTEREST FOR A GIVEN PAIR OF WORDS
def features_overlap_lag3(iComb, patts, nFeat, c_sum = True):
    match = []
    for iFeat in range(nFeat):
        if any(i == iComb[iFeat] for i in patts):
            match.append(1)
        else:
            match.append(0)
    if c_sum:
        return sum(match)
    else:
        return match

# COMPUTE FEATURES OVERLAP FOR EACH PAIR OF WORDS
def compute_feats_overlap(Words, Feats, patts, nFeats):
    Ovlap = []
    for idx_1 in range(len(Words)):
        ovlap = []
        for idx_2 in range(len(Words)):
            iComb = [tuple(i + j) for i, j in zip(Feats[idx_1], Feats[idx_2])]
            ovlap.append(features_overlap_lag3(iComb, patts, nFeat=nFeats))
        Ovlap.append(ovlap)
    return Ovlap

# COMPUTE THEORETICAL RHYTHMICITY INDEX AT THE FREQUENCY OF INTEREST FOR A SET OF FOUR WORDS
def theoretical_feats_idx(set_i, feats, patts):
    ov_theory = []
    for idx_A in range(nTrip):
        for idx_B in range(nTrip):
            iComb = [tuple(i + j) for i, j in
                      zip(feats[idx_A], feats[idx_B])]
            if idx_A != idx_B:
                ovlap = features_overlap_lag3(iComb, patts, False)
                ovlap = [i for i in range(nFeat) if ovlap[i] == 1]
                ov_theory.append(ovlap)
    ov_theory = [i for i in ov_theory if i]
    ov_theory = list(it.chain.from_iterable(ov_theory))
    ov_theory = cnt(ov_theory)
    ov_theory = [int(ov_theory[i] / 2) for i in range(nFeat)]
    if sum(ov_theory) <= nPoss * 2 and not any(i > 1 for i in ov_theory):
        ri_theory = []
        perms = list(it.permutations(feats))
        for iPerm in perms:
            iComb = [tuple(i + j + k + m) for i, j, k, m in
                     zip(iPerm[0], iPerm[1], iPerm[2], iPerm[3])]
            count = []
            for iFeat in range(nFeat):
                c = 0
                for iSyll in range(len(iComb[iFeat]) - nPoss * 2):
                    part_v = iComb[iFeat][iSyll : iSyll + nPoss * 2]
                    if any(i_Pat == part_v for i_Pat in patts):
                        c += 1
                count.append(c / (len(iComb[iFeat]) - nPoss * 2))
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

# !!! GENERATE SETS OF WORDS WITH UNIQUE SYLLABLES AND MINIMUM FEATURES OVERLAP
def generate_lexicon_sets(Words, Sylls, Feats, Ovlap):
    nWord = len(Words)
    set_X = set()
    for idx_1 in tqdm(range(nWord)):
        syll1 = Sylls[idx_1]
        set_1 = list(range(nWord))
        set_2 = [i for i in set_1 if not any(iSyll in Words[i] for iSyll in syll1)]
        set_2 = [i for i in set_2 if Ovlap[idx_1][i] <= 1]
        set_x = [[idx_1, idx_2] for idx_2 in set_2]
        set_x = set(frozenset(i) for i in set_x)
        set_X.update(set_x)
    set_n = [list(i) for i in set_X]
    set_2 = [i for i in set([set_i[-1] for set_i in set_n])]
    set_X = set()
    for set_i in tqdm(set_n):
        idx_1 = set_i[0]
        idx_2 = set_i[1]
        syll1 = Sylls[idx_1]
        syll2 = Sylls[idx_2]
        sylls = syll1 + syll2
        set_3 = [i for i in set_2 if not any(iSyll in Words[i] for iSyll in sylls)]
        set_3 = [i for i in set_3 if Ovlap[idx_1][i] <= 1 and Ovlap[idx_2][i] <= 1]
        set_x = [[idx_1, idx_2, idx_3] for idx_3 in set_3]
        set_x = set(frozenset(i) for i in set_x)
        set_X.update(set_x)
    set_n = [list(i) for i in set_X]
    set_3 = [i for i in set([set_i[-1] for set_i in set_n])]
    Sets_of_Words = []; Sets_of_Feats = []
    for set_i in tqdm(set_n):
        idx_1 = set_i[0]
        idx_2 = set_i[1]
        idx_3 = set_i[2]
        syll1 = Sylls[idx_1]
        syll2 = Sylls[idx_2]
        syll3 = Sylls[idx_3]
        sylls = syll1 + syll2 + syll3
        set_4 = [i for i in set_3 if not any(iSyll in Words[i] for iSyll in sylls)]
        set_4 = [i for i in set_4
                  if Ovlap[idx_1][i] <= 1 and Ovlap[idx_2][i] <= 1 and Ovlap[idx_3][i] <= 1]
        set_x = [[idx_1, idx_2, idx_3, idx_4] for idx_4 in set_4]
        for set_I in set_x:
            feats = [Feats[i] for i in set_I]
            set_X, ov_theory, ri_theory = theoretical_feats_idx(set_I, feats, patts)
            if set_X:
                Sets_of_Words.append([Words[i] for i in set_X])
                Sets_of_Feats.append([ov_theory, ri_theory])
    return Sets_of_Words, Sets_of_Feats

# COMPUTE TRANSITIONAL PROBABILITY MATRIX (M) GIVEN A SEQUENCE (V)
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
def pseudo_walk_TP_random(P, T, v, S, N):
    t = []
    for iTrip in range(nTrip):
        for iPoss in range(nPoss):
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

# GENERATE POSITION-CONTROLLED BASELINE WITH UNIFORM TPs
def pseudo_rand_TP_random():
    P = [list(range(i,nSyll,nPoss)) for i in range(nPoss)]
    I = list(range(nPoss))
    B = I[:]
    B.append(B.pop(0))
    T = [[list(i) for i in list(it.product(*[P[j], P[k]]))] for j, k in zip(I, B)]
    while True:
        V = []
        while len(V) < nSyll*nReps:
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
                if len(t) < nSyll:
                    v = []
                    S = []
                    t = []
            V += v
        V.append(V[0])
        M = np.array(transitional_p_matrix(V))
        V.pop()
        if all(set(i) <= set([1/nTrip, 0]) for i in M):
            break
    return V, M

# GENERATE RANDOMIZATIONS CONTROLLING FOR UNIFORMITY OF TPs ACROSS WORDS
def pseudo_rand_TP_struct():
    while True:
        P = list(it.permutations(np.arange(nTrip)))
        S = [[*(i for _ in range(math.ceil(nReps / len(P))))] for i in P]
        M = np.zeros((nTrip, nTrip))
        v = []
        c = 2
        mask = (np.ones((nTrip, nTrip)) - np.diag(np.ones(nTrip))).astype(bool)
        for _ in range(nReps):
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
                T = [i for i in P
                      if any(i for i in [((s, i)) for i in n]) in bytes(i) and i[0] != s]
                t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                if not t:
                    t = T
                while True:
                    if cnt(M[mask])[c - 1] >= len(M[mask]) - nTrip:
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
                        T = [i for i in P
                              if any(i for i in [((s, i)) for i in n]) in bytes(i) and i[0] != s]
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
        if all(np.diagonal(M) == np.zeros(nTrip)) and len(v) == nTrip * nReps:
            if M[mask].min() == math.floor(nReps * nTrip / len(M[mask])) - 1:
                if M[mask].max() == math.ceil(nReps * nTrip / len(M[mask])):
                    v.append(v[0])
                    M = np.array(transitional_p_matrix(v))
                    v.pop()
                    break
    return v, M

# COMPUTE EMPIRICAL RHYTHMICITY INDEX AT THE FREQUENCY OF INTEREST FOR A SEQUENCE OF SYLLABLES
def features_stationarity(trial, patts):
    cnsnt = [i[0] for i in trial]
    vowel = [i[1:] for i in trial]
    stationarity_index = []
    for iChar in [cnsnt, vowel]:
        feats = [numbs[phons.index(i)] for i in iChar]
        count_patterns = []
        if iChar == cnsnt:
            nFeat = [i for i in range(len(labls)) if labls[i] in lbl_C]
        else:
            nFeat = [i for i in range(len(labls)) if labls[i] in lbl_V]
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
    stationarity_index = list(it.chain.from_iterable(stationarity_index))
    return stationarity_index


def gen_lexicons():
    # %%#################################################################################################
    ############################################# LEXICONS #############################################
    ####################################################################################################
    lbl_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
    lbl_V = ['back', 'hi', 'lo', 'lab', 'tense']
    nFeat = len(lbl_C) + len(lbl_V)

    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)
    numbs = bin_feat.numbs
    phons = bin_feat.phons
    labls = bin_feat.labels

    # # LOAD WORDS
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'words.pickle')
    with open(fname, 'rb') as f:
        fdata = pickle.load(f)
    Words = fdata[0]; Trips = fdata[1]
    print(Trips)
    exit()

    # KERNELS SIMULATING A STATIONARY OSCILLATORY SIGNAL AT THE FREQUENCY OF INTEREST (LAG = 3)
    patts = [tuple([i for i in np.roll((1, 0, 0, 1, 0, 0), j)]) for j in range(nPoss)]

    # EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR
    Sylls = [[w[i:j] for i, j in zip(list(range(0, nPoss * nPoss, nPoss)),
                                     list(range(0, nPoss * nPoss, nPoss))[1:] + [None])]
             for w in Words]
    Feats = [binary_feature_matrix(i, numbs, phons, labls) for i in Sylls]
    Ovlap = compute_feats_overlap(Words, Feats, patts, nFeats=nFeat)

    # GENERATE SETS OF WORDS WITH UNIQUE SYLLABLES AND MINIMUM FEATURES OVERLAP ACROSS WORDS
    Sets_of_Words, Sets_of_Feats = generate_lexicon_sets(Words, Sylls, Feats, Ovlap)
    exit()

    # # SAVE LEXICONS
    # opdir = project_dir + '01_Stimuli/03_Lexicons/'
    # fname = opdir + 'lexicons.pickle'
    # fdata = [Sets_of_Words, Sets_of_Feats]
    # with open(fname, 'wb') as f:
    #     pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)

    # %%#################################################################################################
    ############################################## STREAM ##############################################
    ####################################################################################################

    # # LOAD LEXICONS
    # indir = project_dir + '01_Stimuli/03_Lexicons/'
    # fname = indir + 'lexicons.pickle'
    # with open(fname, 'rb') as f:
    #     fdata = pickle.load(f)
    # Sets_of_Words = fdata[0]; Sets_of_Feats = fdata[1]

    # # GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs
    # TP_struct_V = []; TP_random_V = []
    # while len(TP_struct_V) < nRand:
    #     v_struct, M_struct = pseudo_rand_TP_struct()
    #     v_random, M_struct = pseudo_rand_TP_random()
    #     if v_struct not in TP_struct_V and v_random not in TP_random_V:
    #         TP_struct_V.append(v_struct)
    #         TP_random_V.append(v_random)

    # # SAVE RANDOMIZATIONS (nRand = 100000, nReps = 48)
    # opdir = project_dir + '01_Stimuli/04_Streams/'
    # fname = opdir + 'pseudo_rand.pickle'
    # fdata = [TP_struct_V, TP_random_V]
    # with open(fname, 'wb') as f:
    #     pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)

    # LOAD RANDOMIZATIONS (nRand = 100000, nReps = 48)
    indir = project_dir + '01_Stimuli/04_Streams/'
    fname = indir + 'pseudo_rand.pickle'
    with open(fname, 'rb') as f:
        fdata = pickle.load(f)
    TP_struct_V = fdata[0];
    TP_random_V = fdata[1]

    # SELECT TP STRUCT STREAMS AND TP RANDOM STREAMS WITH MINIMUM RHYTHMICITY INDEX
    TP_struct_trial = [];
    RI_struct_trial = []
    TP_random_trial = [];
    RI_random_trial = []
    Rhythmicity_max = []
    for iSets in range(len(Sets_of_Words)):
        print(iSets)
        tp_struct_trl = [];
        ri_struct_trl = []
        tp_random_trl = [];
        ri_random_trl = []
        words = Sets_of_Words[iSets][:]
        sylls = list(it.chain.from_iterable(
            [[w[i:j] for i, j in zip(list(range(0, nPoss * nPoss, nPoss)),
                                     list(range(0, nPoss * nPoss, nPoss))[1:] + [None])]
             for w in words]))
        for iRand in range(nRand):
            tp_struct = list(it.chain.from_iterable(
                [[w[i:j] for i, j in zip(list(range(0, nPoss * nPoss, nPoss)),
                                         list(range(0, nPoss * nPoss, nPoss))[1:] + [None])]
                 for w in [words[i] for i in TP_struct_V[iRand]]]))
            tp_random = [sylls[i] for i in TP_random_V[iRand]]
            ri_struct = features_stationarity(tp_struct, patts)
            ri_random = features_stationarity(tp_random, patts)
            if max(ri_struct) <= 0.1:
                tp_struct_trl.append(tp_struct)
                ri_struct_trl.append(ri_struct)
            if max(ri_random) <= 0.1:
                tp_random_trl.append(tp_random)
                ri_random_trl.append(ri_random)
        tp_struct_idx = np.argsort([max(i) for i in ri_struct_trl])[:nTrls]
        TP_struct_trl = [tp_struct_trl[i] for i in tp_struct_idx]
        RI_struct_trl = [ri_struct_trl[i] for i in tp_struct_idx]
        tp_random_idx = np.argsort([max(i) for i in ri_random_trl])[:nTrls]
        TP_random_trl = [tp_random_trl[i] for i in tp_random_idx]
        RI_random_trl = [ri_random_trl[i] for i in tp_random_idx]
        ri_struct_max = max(max(RI_struct_trl))
        ri_random_max = max(max(RI_struct_trl))
        rhythmicity_max = max([ri_struct_max, ri_random_max])
        TP_struct_trial.append(TP_struct_trl)
        TP_random_trial.append(TP_random_trl)
        RI_struct_trial.append(RI_struct_trl)
        RI_random_trial.append(RI_random_trl)
        Rhythmicity_max.append(rhythmicity_max)
    best_RI = np.argsort(Rhythmicity_max)[:nSets]
    TP_struct_stream = [TP_struct_trial[i] for i in best_RI]
    TP_random_stream = [TP_random_trial[i] for i in best_RI]
    RI_struct_stream = [RI_struct_trial[i] for i in best_RI]
    RI_random_stream = [RI_random_trial[i] for i in best_RI]

    # SAVE LEXICON TRIALS
    opdir = project_dir + '01_Stimuli/04_Streams/'
    fname = indir + 'streams.pickle'
    fdata = [TP_struct_stream, TP_random_stream, RI_struct_stream, RI_random_stream]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    gen_lexicons()
