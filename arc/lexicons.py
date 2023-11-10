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


def sample_min_overlap_lexicon(words, overlap, n_words=6, max_overlap=1, max_yields=10):
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

        def check_syll_pair(p):
            li = [s.syll for i in p for s in words[i]]
            return len(set(li)) == len(li)

        valid_pairs = set(filter(check_syll_pair, valid_pairs))

        for start_pair in valid_pairs:
            # print(f"max overlap: {max_overlap}; start with pair: {i}/{len(valid_pairs)}")
            lexicon_indexes = set(start_pair)

            for candidate_idx in range(len(overlap)):
                if candidate_idx not in lexicon_indexes:
                    has_min_overlap = [({known_idx, candidate_idx} in valid_pairs) for known_idx in lexicon_indexes]

                    if all(has_min_overlap):
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


def sample_word_randomization(lexicon: Lexicon, randomized_word_indexes):
    for iRand in range(N_RANDOMIZATIONS_PER_STREAM):

        rand_words_lexicon = [list(lexicon)[i] for i in randomized_word_indexes[iRand]]
        stream_word_randomized = [s.syll for word in rand_words_lexicon for s in word]

        yield stream_word_randomized


def sample_syllable_randomization(lexicon: Lexicon, randomized_syllable_indexes):
    lexicon_syllables = [s.syll for word in lexicon for s in word]

    for iRand in range(N_RANDOMIZATIONS_PER_STREAM):
        stream_syllable_randomized = [lexicon_syllables[i] for i in randomized_syllable_indexes[iRand]]

        yield stream_syllable_randomized


def check_rhythmicity(stream, patterns, feats, max_ri=0.2):
    rhythmicity_index = compute_rhythmicity_index(stream, patterns, feats)
    if max(rhythmicity_index) <= max_ri:
        return stream, rhythmicity_index

    return None


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
    MAX_RI = 0.5

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=10000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=10000)

    s1w = None
    s1s = None
    s2w = None
    s2s = None

    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.syll for lexicon in [lexicon_1, lexicon_2] for word in lexicon for s in word]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", extract_lexicon_string(lexicon_1), extract_lexicon_string(lexicon_2))

        # generate good-RI streams
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat)
            if s1w:
                break

        if not s1w:
            for stream in [s1w, s1s, s2w, s2s]:
                print(stream)
            continue

        for stream_2_word_randomized in sample_word_randomization(lexicon_2, randomized_word_indexes):
            s2w = check_rhythmicity(stream_2_word_randomized, oscillation_patterns, bin_feat)
            if s2w:
                break

        if not s2w:
            for stream in [s1w, s1s, s2w, s2s]:
                print(stream)
            continue

        for stream_1_syll_randomized in sample_syllable_randomization(lexicon_1, randomized_syllable_indexes):
            s1s = check_rhythmicity(stream_1_syll_randomized, oscillation_patterns, bin_feat)
            if s1s:
                break

        if not s1s:
            for stream in [s1w, s1s, s2w, s2s]:
                print(stream)
            continue

        for stream_2_syll_randomized in sample_syllable_randomization(lexicon_2, randomized_syllable_indexes):
            s2s = check_rhythmicity(stream_2_syll_randomized, oscillation_patterns, bin_feat)
            if s2s:
                break

        if all([s1w, s1s, s2w, s2s]):
            break

    for stream in [s1w, s1s, s2w, s2s]:
        print(stream)

    # SAVE LEXICON TRIALS
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'streams.pickle')
    fdata = [s1w, s1s, s2w, s2s]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    gen_streams()
