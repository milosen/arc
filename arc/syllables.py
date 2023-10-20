import csv
import os
import pickle
from dataclasses import dataclass
from itertools import groupby
from pprint import pprint
from typing import Tuple, NamedTuple, List

import numpy as np
from scipy import stats
from tqdm.rich import tqdm

from arc import CORPUS_DEFAULT_PATH, RESULTS_DEFAULT_PATH
from arc.phonecodes import phonecodes


SYLLABLES_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'syll.txt')
BINARY_FEATURES_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'binary_features.csv')
IPA_BIGRAMS_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'ipa_bigrams_german.csv')
IPA_TRIGRAMS_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'ipa_trigrams_german.csv')
IPA_SEG_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'german_IPA_seg.csv')

LABELS_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
LABELS_V = ['back', 'hi', 'lo', 'lab', 'tense']


@dataclass
class Phoneme:
    phon: str
    order: int


@dataclass
class Syllable:
    syll: str
    freq: int
    prob: float
    p_unif: float


@dataclass
class Ngram:
    ngram: str
    freq: float
    p_unif: float


PhonemesList = List[Phoneme]
SyllablesList = List[Syllable]
NgramsList = List[Ngram]


@dataclass
class BinaryFeatures:
    labels: List[str]
    labels_c: List[str]
    labels_v: List[str]
    phons: List[str]
    numbs: List[str]
    consonants: List[str]
    long_vowels: List[str]
    n_features: int

    def print_value(self, value):
        s = ""
        if isinstance(value, list):
            s += "["
            for v in value[:3]:
                s += str(self.print_value(v))
                s += ", "
            s += "...]"
        else:
            s += str(value)
        return s

    def __str__(self):
        s = ""
        for key, value in self.__dict__.items():
            s += key
            s += ": "
            s += self.print_value(value)
            s += ", "
        s = s[:-2]
        return f"BinaryFeatures({s})"


class SyllablesData(NamedTuple):
    sylls: List
    phons_rare: List
    bigrams: List
    trigrams: List
    bigrams_prob_filtered: List
    trigrams_prob_filtered: List


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


def read_order_of_phonemes(ipa_seg_path: str = IPA_SEG_DEFAULT_PATH) -> PhonemesList:
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


def select_consonant_vowel(
        syllables: SyllablesList,
        bin_feats: BinaryFeatures,
        single_onset_consonant: bool = True,
        vowel_mode: str = "long"
) -> SyllablesList:
    """Only select those syllables that follow the pattern: (single-phoneme) consonant followed by a (long) vowel"""
    print("SELECT CONSONANT-VOWEL SYLLABLES WITH LONG VOWEL LENGTH")

    consonants = bin_feats.consonants
    if single_onset_consonant:
        consonants = list(filter(lambda x: len(x) == 1, consonants))

    # TODO: implement other vowel modes
    if vowel_mode == "long":
        vowels = bin_feats.long_vowels
    elif vowel_mode in ["short", "any"]:
        raise NotImplementedError(f"Vowel mode {vowel_mode} not supported yet.")
    else:
        raise KeyError("Unknown vowel mode.")

    prefixes = tuple(consonants)
    postfixes = tuple(vowels)

    syllables = list(filter(
        lambda s: s.syll.startswith(prefixes) and s.syll.endswith(postfixes), syllables)
    )

    return syllables


def select_occurrence_probability(syllables: SyllablesList, p_threshold: float = 0.05) -> SyllablesList:
    print("SELECT CV SYLLABLES WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")

    syllables = list(filter(
        lambda s: s.p_unif > p_threshold, syllables
    ))

    return syllables


def select_native_phonemes(syllables: SyllablesList, phonemes: PhonemesList) -> SyllablesList:
    print("REMOVE SYLLABLES WITH NON-NATIVE PHONEMES")

    prefixes = tuple(phoneme.phon for phoneme in phonemes)

    syllables = list(filter(
        lambda s: s.syll.startswith(prefixes), syllables
    ))

    return syllables


def grams_filter_uniform(n_grams: NgramsList, p_threshold: float = 0.05) -> NgramsList:
    ngrams = list(filter(
        lambda s: s.p_unif > p_threshold, n_grams
    ))

    return ngrams


def filter_rare_onset_phonemes(syllables: SyllablesList, phonemes: PhonemesList,
                               p_threshold: float = 0.05) -> PhonemesList:
    print("FIND SYLLABLES THAT ARE RARE AT THE ONSET OF A WORD")
    candidate_onset_phonemes = set([syll.syll[0] for syll in syllables])

    onset_phonemes = [x.phon for x in phonemes if x.order == 1]
    all_phonemes = [x.phon for x in phonemes]

    rare_phonemes = set()
    for phon in tqdm(candidate_onset_phonemes):
        phoneme_prob = onset_phonemes.count(phon) / all_phonemes.count(phon)
        if phoneme_prob < p_threshold:
            rare_phonemes.add(phon)
    rare_phonemes = list(rare_phonemes)
    return rare_phonemes


def export_speech_synthesiser(syllables: SyllablesList):
    print("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
    syllables_dir = os.path.join(RESULTS_DEFAULT_PATH, "syllables")
    os.makedirs(syllables_dir, exist_ok=True)
    c = [i.syll[0] for i in syllables]
    v = [i.syll[1] for i in syllables]
    c = ' '.join(c).replace('ʃ', 'sch').replace('ɡ', 'g').replace('ç', 'ch').replace('ʒ', 'dsch').split()
    v = ' '.join(v).replace('ɛ', 'ä').replace('ø', 'ö').replace('y', 'ü').split()
    t = [c[i] + v[i] for i in range(len(syllables))]
    for syllable, text in zip(syllables, t):
        synth_string = '<phoneme alphabet="ipa" ph=' + '"' + syllable.syll + '"' + '>' + text + '</phoneme>'
        with open(os.path.join(syllables_dir, f'{str(syllable.syll[0:2])}.txt'), 'w') as f:
            f.write(synth_string + "\n")
            csv.writer(f)


def generate_syllables():
    syllables = read_syllables()
    print(syllables[:10])

    bin_feats = read_binary_features()
    print(bin_feats)

    phonemes = read_order_of_phonemes()
    print(phonemes[:10])

    bigrams = read_bigrams()
    print(bigrams[:10])

    trigrams = read_trigrams()
    print(trigrams[:10])

    syllables = select_consonant_vowel(syllables, bin_feats)
    print(syllables[:10])

    syllables = select_occurrence_probability(syllables)
    print(syllables[:10])

    syllables = select_native_phonemes(syllables, phonemes)
    print(syllables[:10])

    print("SELECT BIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    bigrams_uniform = grams_filter_uniform(bigrams)
    print(bigrams_uniform[:10])

    print("SELECT TRIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    trigrams_uniform = grams_filter_uniform(trigrams)
    print(trigrams_uniform[:10])

    rare_onset_phonemes = filter_rare_onset_phonemes(syllables, phonemes)
    print(rare_onset_phonemes[:10])

    print("SAVE SYLLABLES")
    syllables_data = SyllablesData(
        sylls=syllables,
        phons_rare=rare_onset_phonemes,
        bigrams=bigrams,
        trigrams=trigrams,
        bigrams_prob_filtered=bigrams_uniform,
        trigrams_prob_filtered=trigrams_uniform
    )

    with open(os.path.join(RESULTS_DEFAULT_PATH, "syllables.pickle"), 'wb') as f:
        pickle.dump(syllables_data, f, pickle.HIGHEST_PROTOCOL)

    export_speech_synthesiser(syllables)


if __name__ == '__main__':
    generate_syllables()
