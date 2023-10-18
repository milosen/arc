import csv
import os
import pickle
import random
from typing import List

import numpy as np
from scipy.stats import stats
from tqdm.rich import tqdm

from arc.phonecodes import phonecodes
from arc import CORPUS_DEFAULT_PATH, RESULTS_DEFAULT_PATH
from arc.syllables import SYLLABLES_DEFAULT_PATH, BINARY_FEATURES_DEFAULT_PATH, read_binary_features, SyllablesData, \
    Phonemes, print_obj


def generate_subset_sylls(iSyll, allIdx, f_Cls):
    """GENERATE A SUBSET OF SYLLABLES WITH NO OVERLAP OF CLASSES OF FEATURES WITH PREVIOUS SYLLABLES"""
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


def generate_trisyll_word(syllables, f_cls):
    allIdx = set([syllables.index(i) for i in syllables])
    while True:
        syl_1 = syllables.index(random.sample(syllables, 1)[0])
        set_1 = generate_subset_sylls(syl_1, allIdx, f_cls)
        if set_1:
            syl_2 = random.sample(list(set_1), 1)[0]
            set_2 = generate_subset_sylls(syl_2, allIdx, f_cls)
            if set_2:
                set_3 = set_1.intersection(set_1, set_2)
                if set_3:
                    syl_3 = random.sample(list(set_3), 1)[0]
                    break
    return syllables[syl_1] + syllables[syl_2] + syllables[syl_3]


# CHECK IF A TRIPLET HAS NON-ZERO (gram) OR UNIFORM (Gram) BIGRAM AND TRIGRAM LOG-PROBABILITY
def get_corpus_gram_stats(iTrip, Gram2, Gram3):
    """TODO: Change implementation for other syllables-settings?"""
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


def generate_nsyll_words(syllables, f_cls, n_sylls=3, n_tries=1_000_000) -> List:
    words = set()
    all_idx_list = range(len(syllables))
    for _ in tqdm(range(n_tries)):
        word = ""
        idx_list = all_idx_list
        for _ in range(n_sylls):
            if not idx_list:
                break
            idx = random.choice(idx_list)
            word += syllables[idx]
            idx_set = generate_subset_sylls(idx, set(idx_list), f_cls)
            idx_list = list(idx_set)
        words.add(word)
    return words


def generate_words():
    print("LOAD SYLLABLES")
    with open(os.path.join(RESULTS_DEFAULT_PATH, "syllables.pickle"), 'rb') as f:
        syllables_data: SyllablesData = pickle.load(f)

    print_obj(syllables_data)
    exit()

    CVsyl = syllables_data.sylls
    xPhon = syllables_data.phons_rare
    gram2 = syllables_data.bigrams
    Gram2 = syllables_data.bigrams_prob_filtered
    gram3 = syllables_data.trigrams
    Gram3 = syllables_data.trigrams_prob_filtered

    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)
    numbs = bin_feat.numbs
    phons = bin_feat.phons
    labls = bin_feat.labels

    print("COMPUTE MULTI-FEATURE COMBINATIONS FOR EACH CONSONANT (MANNER AND PLACE) AND VOWEL (IDENTITY)")
    i_Son = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('son')] == '+']
    i_Plo = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('son')] != '+'
             and numbs[phons.index(CVsyl[i][0])][labls.index('cont')] != '+']
    i_Fri = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('son')] != '+'
             and numbs[phons.index(CVsyl[i][0])][labls.index('cont')] == '+']
    i_Lab = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('lab')] == '+']
    i_Den = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('cor')] == '+'
             and numbs[phons.index(CVsyl[i][0])][labls.index('hi')] != '+']
    i_Oth = [i for i in range(len(CVsyl)) if i not in i_Lab and i not in i_Den]
    idx_A = [i for i in range(len(CVsyl)) if 'a' in CVsyl[i][1:]]
    idx_E = [i for i in range(len(CVsyl)) if 'e' in CVsyl[i][1:]]
    idx_I = [i for i in range(len(CVsyl)) if 'i' in CVsyl[i][1:]]
    idx_O = [i for i in range(len(CVsyl)) if 'o' in CVsyl[i][1:]]
    idx_U = [i for i in range(len(CVsyl)) if 'u' in CVsyl[i][1:]]
    idx_Ä = [i for i in range(len(CVsyl)) if 'ɛ' in CVsyl[i][1:]]
    idx_Ö = [i for i in range(len(CVsyl)) if 'ø' in CVsyl[i][1:]]
    idx_Ü = [i for i in range(len(CVsyl)) if 'y' in CVsyl[i][1:]]
    f_Mnr = [set(i_Son), set(i_Plo), set(i_Fri)]
    f_Plc = [set(i_Oth), set(i_Lab), set(i_Den)]
    f_Vow = [set(idx_A), set(idx_E), set(idx_I), set(idx_O), set(idx_U),
             set(idx_Ä), set(idx_Ö), set(idx_Ü)]
    f_Cls = [f_Mnr, f_Plc, f_Vow]

    RELOAD_WORDS = False

    if not os.path.exists(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv')) or RELOAD_WORDS:
        print("GENERATE LIST OF TRISYLLABIC WORDS WITH NO OVERLAP OF COMPLEX PHONETIC FEATURES ACROSS SYLLABLES")
        # words = list(set([generate_trisyll_word(CVsyl, f_Cls) for _ in tqdm(range(1000))]))
        words = generate_nsyll_words(CVsyl, f_Cls)
        print("Words generated: ", len(words))

        # SAVE LIST OF TRIPLETS IN ONE CSV FILE
        with open(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv'), 'w') as f:
            w = csv.writer(f)
            for word in words:
                w.writerows([[word, 0]])
    else:
        print("LOAD WORDS")

    # TO ENSURE THAT THE TRIPLETS CANNOT BE MISTAKEN FOR GERMAN WORDS,
    # WE INSTRUCTED A NATIVE GERMAN SPEAKER TO MARK EACH TRIPLET AS...
    #     '1' IF IT CORRESPONDS EXACTLY TO A GERMAN WORD
    #     '2' IF IT COULD BE MISTAKEN FOR A GERMAN WORD-GROUP WHEN PRONOUNCED ALOUD
    #     '3' IF THE PRONOUNCIATION OF THE FIRST TWO SYLLABLES IS A WORD CANDIDATE,
    #         i.e. the syllable pair could be mistaken for a German word or
    #         it evokes a strong prediction for a certain real German word
    #     '4' IF IT DOES NOT SOUND GERMAN AT ALL
    #         (that is, if the phoneme combination is illegal in German morphology
    #         [do not flag if rule exceptions exist])
    #     '0' OTHERWISE (that is, the item is good)

    print("LOAD LIST OF TRIPLETS AND SELECT THOSE THAT CANNOT BE MISTAKEN FOR GERMAN WORDS")
    fdata = list(csv.reader(open(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv'), "r"), delimiter='\t'))

    trips = [i[0] for i in [i[0].split(",") for i in fdata] if int(i[1]) == 0]

    print("EXCLUDE WORDS WITH LOW ONSET SYLLABLE PROBABILITY")
    trips = [i for i in trips if i[0] not in xPhon]

    print("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    print(trips)
    exit()
    words = [i for i in tqdm(trips) if get_corpus_gram_stats(i, Gram2, Gram3)]
    print(len(words), len(trips))

    print("SAVE WORDS")
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'words.pickle')
    fdata = [words, trips]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    generate_words()
