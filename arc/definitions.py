import os


CORPUS_DEFAULT_PATH = os.path.join('resources', 'example_corpus')
RESULTS_DEFAULT_PATH = "arc_results"

SYLLABLES_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'syll.txt')
BINARY_FEATURES_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'binary_features.csv')
IPA_BIGRAMS_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'ipa_bigrams_german.csv')
IPA_TRIGRAMS_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'ipa_trigrams_german.csv')
IPA_SEG_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'german_IPA_seg.csv')

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
