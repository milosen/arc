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
import itertools
import random

# PROJECT DIRECTORY
project_dir = '/'

#%%#################################################################################################
############################################## IMPORT ##############################################
####################################################################################################

import pickle

from arc.functional import binary_feature_matrix, compute_word_overlap_matrix, map_iterable, pseudo_rand_TP_struct, \
    pseudo_rand_TP_random, compute_rhythmicity_index
from arc.io import *
from arc.definitions import *
from arc.types import *


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


def sample_min_overlap_lexicon(words, overlap, n_words=6, max_overlap=1, max_yields=100):
    overlap = np.array(overlap)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    print(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0
    for sup_overlap in range(max_overlap + 1):
        if sup_overlap != 0:
            print(f"Warning: Increasing allowed overlap to {sup_overlap}")

        # WORDSxWORDS boolean matrix indicating if the words can be paired together
        # e.g. valid_word_pairs_matrix[0, 0] = False, bc. no word is parable with itself
        valid_word_pairs_matrix = (overlap <= sup_overlap)

        # represent the matrix from above as a list of pairs of word indexes, e.g. [[0, 1], [0, 2], ...]
        valid_pairs = set(frozenset(i) for i in zip(*np.where(valid_word_pairs_matrix)))

        for start_pair in valid_pairs:
            # print(f"max overlap: {max_overlap}; start with pair: {i}/{len(valid_pairs)}")
            lexicon_indexes = set(start_pair)

            for candidate_idx in range(len(overlap)):
                if candidate_idx not in lexicon_indexes:
                    fits_with_known = [({known_idx, candidate_idx} in valid_pairs) for known_idx in lexicon_indexes]

                    if all(fits_with_known):
                        lexicon_indexes.add(candidate_idx)

                        if len(lexicon_indexes) == n_words:
                            valid_sub_matrix = valid_word_pairs_matrix[np.ix_(list(lexicon_indexes), list(lexicon_indexes))]
                            assert np.all(np.int32(valid_sub_matrix) == (1 - np.eye(n_words)))
                            lexicon = set(map(lambda index: words[index], lexicon_indexes))

                            yield lexicon

                            yields += 1

                            if yields == max_yields:
                                return


def extract_lexicon_string(lexicon: Lexicon) -> str:
    return "".join(s.syll for word in lexicon for s in word)


def gen_streams():
    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)

    # # LOAD WORDS
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'words.pickle')
    with open(fname, 'rb') as f:
        fdata = pickle.load(f)
        words = fdata[0]

    # KERNELS SIMULATING A STATIONARY OSCILLATORY SIGNAL AT THE FREQUENCY OF INTEREST (LAG = 3)
    oscillation_patterns = [tuple([i for i in np.roll((1, 0, 0, 1, 0, 0), j)]) for j in range(N_SYLLABLES_PER_WORD)]

    # EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR
    features = map_iterable(function=lambda word: binary_feature_matrix(word, bin_feat), iterable=words)
    overlap = compute_word_overlap_matrix(words=words, features=features, oscillation_patterns=oscillation_patterns)

    REGENERATE_STREAM_RANDOMIZATION = False
    if not os.path.exists(os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle')) or REGENERATE_STREAM_RANDOMIZATION:
        # GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs
        TP_struct_V = []; TP_random_V = []
        while len(TP_struct_V) < N_RANDOMIZATIONS_PER_STREAM:
            v_struct, M_struct = pseudo_rand_TP_struct()
            v_random, M_struct = pseudo_rand_TP_random()
            if v_struct not in TP_struct_V and v_random not in TP_random_V:
                TP_struct_V.append(v_struct)
                TP_random_V.append(v_random)

        fdata = [TP_struct_V, TP_random_V]

        with open(os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle'), 'wb') as f:
            pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("SKIPPING RANDOM STREAM GENERATION. LOADING STREAMS FROM FILE")
        with open(os.path.join(RESULTS_DEFAULT_PATH, "random_streams_indexes.pickle"), 'rb') as f:
            fdata = pickle.load(f)
            randomized_word_indexes = fdata[0]
            randomized_syllable_indexes = fdata[1]

    # SELECT TP STRUCT STREAMS AND TP RANDOM STREAMS WITH MINIMUM RHYTHMICITY INDEX
    TP_struct_trial = []
    RI_struct_trial = []
    TP_random_trial = []
    RI_random_trial = []
    Rhythmicity_max = []
    N_TRIES_MINIMAL_LEXICON = 10000
    MAX_RI = 0.2

    for lexicon in sample_min_overlap_lexicon(words, overlap, n_words=N_WORDS_PER_LEXICON, max_overlap=1, max_yields=N_TRIES_MINIMAL_LEXICON):
        tp_struct_trl = []
        ri_struct_trl = []
        tp_random_trl = []
        ri_random_trl = []
        lexicon_syllables = [s.syll for word in lexicon for s in word]
        print(extract_lexicon_string(lexicon))
        for iRand in range(N_RANDOMIZATIONS_PER_STREAM):
            rand_words_lexicon = [list(lexicon)[i] for i in randomized_word_indexes[iRand]]
            stream_word_randomized = [s.syll for word in rand_words_lexicon for s in word]

            stream_syllable_randomized = [lexicon_syllables[i] for i in randomized_syllable_indexes[iRand]]

            ri_word = compute_rhythmicity_index(stream_word_randomized, oscillation_patterns, bin_feat)
            ri_syll = compute_rhythmicity_index(stream_syllable_randomized, oscillation_patterns, bin_feat)

            if max(ri_word) <= MAX_RI:
                tp_struct_trl.append(stream_word_randomized)
                ri_struct_trl.append(ri_word)

            if max(ri_syll) <= MAX_RI:
                tp_random_trl.append(stream_syllable_randomized)
                ri_random_trl.append(ri_syll)

        tp_struct_idx = np.argsort([max(i) for i in ri_struct_trl])[:nTrls]
        TP_struct_trl = [tp_struct_trl[i] for i in tp_struct_idx]
        RI_struct_trl = [ri_struct_trl[i] for i in tp_struct_idx]
        tp_random_idx = np.argsort([max(i) for i in ri_random_trl])[:nTrls]
        TP_random_trl = [tp_random_trl[i] for i in tp_random_idx]
        RI_random_trl = [ri_random_trl[i] for i in tp_random_idx]
        ri_struct_max = max(max(RI_struct_trl, default=[0]))
        ri_random_max = max(max(RI_struct_trl, default=[0]))
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
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'streams.pickle')
    fdata = [TP_struct_stream, TP_random_stream, RI_struct_stream, RI_random_stream]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    gen_streams()
