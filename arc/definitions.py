import os
import pathlib
from importlib import resources as importlib_resources


def get_data_path(fname):
    return importlib_resources.files("arc") / "data" / fname


BINARY_FEATURES_DEFAULT_PATH = get_data_path("phonemes.csv")
PHONEMES_DEFAULT_PATH = get_data_path("phonemes.json")

CORPUS_DEFAULT_PATH = get_data_path("example_corpus")
SYLLABLES_DEFAULT_PATH = CORPUS_DEFAULT_PATH / 'syll.txt'
IPA_BIGRAMS_DEFAULT_PATH = CORPUS_DEFAULT_PATH / 'ipa_bigrams_german.csv'
IPA_TRIGRAMS_DEFAULT_PATH = CORPUS_DEFAULT_PATH / 'ipa_trigrams_german.csv'
IPA_SEG_DEFAULT_PATH = CORPUS_DEFAULT_PATH / 'german_IPA_seg.csv'

RESULTS_DEFAULT_PATH = pathlib.Path("arc_results")
SSML_RESULTS_DEFAULT_PATH = RESULTS_DEFAULT_PATH / "syllables"

PHONEME_FEATURE_LABELS = ["syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor",
                          "distr", "lab", "hi", "lo", "back", "round", "tense", "long"]
SON = PHONEME_FEATURE_LABELS.index('son')
CONT = PHONEME_FEATURE_LABELS.index('cont')
LAB = PHONEME_FEATURE_LABELS.index('lab')
COR = PHONEME_FEATURE_LABELS.index('cor')
HI = PHONEME_FEATURE_LABELS.index('hi')

LABELS_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
LABELS_V = ['back', 'hi', 'lo', 'lab', 'tense']
N_FEAT = len(LABELS_C) + len(LABELS_V)  # 14

# GLOBAL PARAMETERS TO GENERATE TRIPLETS OF SYLLABLES
N_WORDS_PER_LEXICON = 4
N_SYLLABLES_PER_WORD = 3
N_CHARS_PER_SYLLABLE = 3
N_PHONEMES_PER_SYLLABLE = 2
nTrls = 1
nSets = 1
N_SYLLABLES_PER_LEXICON = N_SYLLABLES_PER_WORD * N_WORDS_PER_LEXICON
N_REPETITIONS_PER_TRIAL = N_SYLLABLES_PER_LEXICON * 4
N_RANDOMIZATIONS_PER_STREAM = 100
