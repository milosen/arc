import csv
import itertools
import pickle
import random
from functools import reduce
from math import comb
from typing import Union, Any, Generator, Dict, Iterable

import itertools
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Any, Generator, List, Tuple, Dict, Set

from pydantic import BaseModel, ValidationError, PositiveInt
from scipy import stats
from tqdm.rich import tqdm

from arc.definitions import *
from arc.functional import *
from arc.phonecodes import phonecodes
from arc.types import *


@maybe_load_from_file(path=os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle'), force_redo=False)
def generate_stream_randomization():
    print("GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs")
    TP_struct_V = []
    for _ in tqdm(range(N_RANDOMIZATIONS_PER_STREAM)):
        while True:
            v_struct, m_struct = pseudo_rand_TP_struct()
            if v_struct not in TP_struct_V:
                TP_struct_V.append(v_struct)
                break

    TP_random_V = []
    for _ in tqdm(range(N_RANDOMIZATIONS_PER_STREAM)):
        while True:
            v_random, m_struct = pseudo_rand_TP_random()
            if v_random not in TP_random_V:
                TP_random_V.append(v_random)
                break

    return TP_struct_V, TP_random_V


if __name__ == '__main__':
    feature_syllables = read_feature_syllables("cV", return_as_dict=True)

    sylls = merge_with_corpus(feature_syllables)

    freqs = [s.info["freq"] for s in sylls]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    sylls = [s[0] for s in filter(lambda s: s[1] > 0.05, zip(sylls, p_vals_uniform))]

    native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)
    sylls = list(filter(lambda syll: all([(phon.id in native_phonemes) for phon in syll.phonemes]), sylls))

    export_speech_synthesiser(sylls)

    syllabic_features = syllabic_features_from_list(sylls)

    words: List[Word] = generate_words(sylls)

    words = filter_common_onset_words(tqdm(words), sylls, native_phonemes)
    words = list(filter_gram_stats(tqdm(list(words))))

    print("EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR")
    features = [word.features for word in words]
    print(features)

    print("compute word overlap matrix")
    # KERNELS SIMULATING A STATIONARY OSCILLATORY SIGNAL AT THE FREQUENCY OF INTEREST (LAG = 3)
    overlap = overlap_matrix(words[:200])

    randomized_word_indexes, randomized_syllable_indexes = generate_stream_randomization()

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    s1w = None
    for lexicon_1 in lexicon_generator_1:
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):

            s1w = check_rhythmicity(stream_1_word_randomized, max_ri=0.1)

            if s1w:
                print([s.id for s in s1w[0]], s1w[1])
                exit()
                break

        if not s1w:
            continue

        break

    if s1w:
        stream, info = s1w
        print("Stream 1: ", "".join(syllable for syllable in stream))
        print("Rhythmicity Indexes 1: ", info["rhythmicity_indexes"])
        print("Found lexicon: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"])
    else:
        print("Nothing found")

    exit()

    s1w, s2w = None, None
    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    # print pairwise lexicon generation
    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.id for lexicon in [lexicon_1, lexicon_2] for word in lexicon.words for s in word.syllables]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"], lexicon_2.id,
              lexicon_2.info["cumulative_overlap"])

        s1w = None
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat)

            if s1w:
                break

        if not s1w:
            continue

        s2w = None
        for stream_2_word_randomized in sample_word_randomization(lexicon_2, randomized_word_indexes):
            s2w = check_rhythmicity(stream_2_word_randomized, oscillation_patterns, bin_feat)

            if s2w:
                break

        if not s2w:
            continue

        break

    if s1w and s2w:
        stream1, stream_info1 = s1w
        stream2, stream_info2 = s2w
        print("Stream 1: ", "".join(syllable for syllable in stream1))
        print("Stream 2: ", "".join(syllable for syllable in stream2))
        print("Rhythmicity Indexes 1: ", stream_info1["rhythmicity_indexes"])
        print("Rhythmicity Indexes 2: ", stream_info2["rhythmicity_indexes"])
        print("Found compatible lexicons: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"], lexicon_2.id,
              lexicon_2.info["cumulative_overlap"])
    else:
        print("Nothing found")

    exit()

    print("SELECT TP STRUCT STREAMS AND TP RANDOM STREAMS WITH MINIMUM RHYTHMICITY INDEX")

    s1w = None
    s1s = None
    s2w = None
    s2s = None

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.id for lexicon in [lexicon_1, lexicon_2] for word in lexicon.words for s in word.syllables]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", lexicon_1.id, lexicon_2.id)

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
