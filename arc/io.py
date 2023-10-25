import csv
from typing import Callable, Iterable

import numpy as np
from scipy import stats
from tqdm.rich import tqdm

from arc.phonecodes import phonecodes
from arc.definitions import *
from arc.types import *


def read_syllables(syllables_path: str = SYLLABLES_DEFAULT_PATH) -> SyllablesList:
    print("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")
    fdata = list(csv.reader(open(syllables_path, "r"), delimiter='\t'))
    syllables = []

    freqs = [int(syll_data[2]) for syll_data in fdata[1:]]
    p_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    for syll_data, p_unif in zip(fdata[1:], p_uniform):
        syllables.append(Syllable(syll=phonecodes.xsampa2ipa(syll_data[1], 'deu'),
                                  freq=int(syll_data[2]),
                                  prob=float(syll_data[3]),
                                  p_unif=p_unif))
    return syllables


def read_binary_features(binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH) -> BinaryFeatures:
    print("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")
    fdata = list(csv.reader(open(binary_features_path, "r")))
    labels = fdata[0][1:]
    phons = [i[0] for i in fdata[1:]]
    numbs = [i[1:] for i in fdata[1:]]

    consonants = []
    for phon, numb in zip(phons, numbs):
        if numb[labels.index('cons')] == '+':
            consonants.append(phon)

    long_vowels = []
    for phon, numb in zip(phons, numbs):
        if numb[labels.index('long')] == '+' and phon not in consonants:
            long_vowels.append(phon)

    bin_feats = BinaryFeatures(
        labels=labels,
        labels_c=LABELS_C,
        labels_v=LABELS_V,
        phons=phons,
        numbs=numbs,
        consonants=consonants,
        long_vowels=long_vowels,
        n_features=(len(LABELS_C) + len(LABELS_V))
    )

    return bin_feats


def read_phonemes(ipa_seg_path: str = IPA_SEG_DEFAULT_PATH) -> PhonemesList:
    print("READ ORDER OF PHONEMES IN WORDS")
    fdata = list(csv.reader(open(ipa_seg_path, "r"), delimiter='\t'))
    phonemes = []
    for phon_data in tqdm(fdata[1:]):
        phon_data_split = phon_data[0].split(",")
        if len(phon_data_split) == 3:
            phon = phon_data_split[1].replace('"', '').replace("g", "ɡ")
            order = int(phon_data_split[2])
            phonemes.append(Phoneme(phon=phon, order=order))
    return phonemes


def read_bigrams(ipa_bigrams_path: str = IPA_BIGRAMS_DEFAULT_PATH) -> List[Ngram]:
    print("READ BIGRAMS")
    fdata = list(csv.reader(open(ipa_bigrams_path, "r"), delimiter='\t'))

    freqs = [int(data[0].split(",")[2]) for data in fdata[1:]]
    p_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    bigrams = []
    for data, p_unif in zip(fdata[1:], p_uniform):
        data = data[0].split(",")
        bigram = data[1].replace('_', '').replace("g", "ɡ")
        freq = int(data[2])
        bigrams.append(Ngram(ngram=bigram, freq=freq, p_unif=p_unif))

    return bigrams


def read_trigrams(ipa_trigrams_path: str = IPA_TRIGRAMS_DEFAULT_PATH) -> List[Ngram]:
    print("READ TRIGRAMS")
    fdata = list(csv.reader(open(ipa_trigrams_path, "r"), delimiter='\t'))

    freqs = [int(data[0].split(",")[1]) for data in fdata[1:]]
    p_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    trigrams = []
    for data, p_unif in zip(fdata[1:], p_uniform):
        data = data[0].split(",")
        trigram = data[0].replace('_', '').replace("g", "ɡ")
        freq = int(data[1])
        trigrams.append(Ngram(ngram=trigram, freq=freq, p_unif=p_unif))

    return trigrams
