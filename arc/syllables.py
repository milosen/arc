import csv
import os
import pickle

import numpy as np
from scipy import stats
from tqdm.rich import tqdm

from arc import CORPUS_DEFAULT_PATH, RESULTS_DEFAULT_PATH

import csv
from typing import Tuple, NamedTuple, List

from arc.phonecodes import phonecodes


SYLLABLES_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'syll.txt')
BINARY_FEATURES_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'binary_features.csv')
IPA_BIGRAMS_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'ipa_bigrams_german.csv')
IPA_TRIGRAMS_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'ipa_trigrams_german.csv')
IPA_SEG_DEFAULT_PATH = os.path.join(CORPUS_DEFAULT_PATH, 'german_IPA_seg.csv')


class Syllables(NamedTuple):
    sylls: List
    freqs: List
    probs: List


class BinaryFeatures(NamedTuple):
    labels: List
    phons: List
    numbs: List
    consonants: List
    long_vowels: List


class Phonemes(NamedTuple):
    phons: List
    probs: List


class Ngrams(NamedTuple):
    grams: List
    freqs: List


class Bigrams(Ngrams):
    pass


class Trigrams(Ngrams):
    pass


class SyllablesData(NamedTuple):
    sylls: List
    phons_rare: List
    bigrams: List
    trigrams: List
    bigrams_prob_filtered: List
    trigrams_prob_filtered: List


def read_syllables(syllables_path: str) -> Syllables:
    print("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")
    fdata = list(csv.reader(open(syllables_path, "r"), delimiter='\t'))
    syllables = [phonecodes.xsampa2ipa(i[1], 'deu') for i in fdata[1:]]
    frequencies = [int(i[2]) for i in fdata[1:]]
    probabilities = [float(i[3]) for i in fdata[1:]]
    return Syllables(syllables, frequencies, probabilities)


def read_binary_features(binary_features_path: str) -> BinaryFeatures:
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

    # lbl_C = ['son', 'back', 'hi', 'lab', 'cor', 'cont', 'lat', 'nas', 'voi']
    # lbl_V = ['back', 'hi', 'lo', 'lab', 'tense']
    # nFeat = len(lbl_C) + len(lbl_V)

    return BinaryFeatures(
        labels=labels,
        phons=phons,
        numbs=numbs,
        consonants=consonants,
        long_vowels=long_vowels
    )


def read_order_of_phonemes(ipa_seg_path):
    print("READ ORDER OF PHONEMES IN WORDS")
    fdata = list(csv.reader(open(ipa_seg_path, "r"), delimiter='\t'))
    fdata = [i[0].split(",") for i in fdata][1:]
    fdata = [i for i in fdata if len(i) == 3]  # TODO filter by previously found onset phonemes i == .startswith()
    phon_x = [i[1].replace('"', '') for i in fdata]
    phon_x = [i.replace("g", "ɡ") for i in phon_x]
    phon_p = [int(i[2]) for i in fdata]
    return Phonemes(phons=phon_x, probs=phon_p)


def select_consonant_vowel(
        syllables: Syllables,
        bin_feats: BinaryFeatures,
        single_onset_consonant: bool = True,
        vowel_mode: str = "long"
) -> Syllables:
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

    indexes = []
    for i, syllable in enumerate(syllables.sylls):
        if syllable.startswith(tuple(consonants)) and syllable.endswith(tuple(vowels)):
            indexes.append(i)

    return Syllables(
        sylls=[syllables.sylls[i] for i in indexes],
        freqs=[syllables.freqs[i] for i in indexes],
        probs=[syllables.probs[i] for i in indexes]
    )


def select_occurrence_probability(syllables: Syllables, p_threshold: float = 0.05) -> Syllables:
    print("SELECT CV SYLLABLES WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    log_probs = stats.uniform.sf(abs(stats.zscore(np.log(syllables.freqs))))

    indexes = []
    for i, p in enumerate(log_probs):
        if p > p_threshold:
            indexes.append(i)

    return Syllables(
        sylls=[syllables.sylls[i] for i in indexes],
        freqs=[syllables.freqs[i] for i in indexes],
        probs=[syllables.probs[i] for i in indexes]
    )


def select_native_phonemes(syllables: Syllables, phonemes: Phonemes) -> Syllables:
    print("REMOVE SYLLABLES WITH NON-NATIVE PHONEMES")

    indexes = []
    for i, syll in enumerate(syllables.sylls):
        if syll[0] in phonemes.phons:
            indexes.append(i)

    return Syllables(
        sylls=[syllables.sylls[i] for i in indexes],
        freqs=[syllables.freqs[i] for i in indexes],
        probs=[syllables.probs[i] for i in indexes]
    )


def read_bigrams(ipa_bigrams_path):
    print("READ BIGRAMS")
    gram = list(csv.reader(open(ipa_bigrams_path, "r"), delimiter='\t'))
    gram = [i[0].split(",") for i in gram][1:]
    freq = [int(i[2]) for i in gram]
    gram = [i[1].replace("_", "") for i in gram]
    gram = [i.replace("g", "ɡ") for i in gram]
    return Ngrams(grams=gram, freqs=freq)


def read_trigrams(ipa_trigrams_path):
    print("READ TRIGRAMS")
    gram = list(csv.reader(open(ipa_trigrams_path, "r"), delimiter='\t'))
    gram = [i[0].split(",") for i in gram][1:]
    freq = [int(i[1]) for i in gram]  # different from bigrams !
    gram = [i[0].replace("_", "") for i in gram]
    gram = [i.replace("g", "ɡ") for i in gram]
    return Ngrams(grams=gram, freqs=freq)


def grams_filter_uniform(n_grams: Ngrams, p_threshold: float = 0.05):
    probs = stats.uniform.sf(abs(stats.zscore(np.log(n_grams.freqs))))

    indexes = []
    for i, p in enumerate(probs):
        if p > p_threshold:
            indexes.append(i)

    return Ngrams(grams=[n_grams.grams[i] for i in indexes], freqs=[n_grams.freqs[i] for i in indexes])


def filter_rare_onset_phonemes(syllables: Syllables, phonemes: Phonemes) -> Phonemes:
    print("FIND SYLLABLES THAT ARE RARE AT THE ONSET OF A WORD")
    p_phon = []
    consonants = [i[0] for i in syllables.sylls]
    for i in tqdm(range(len(consonants))):
        phfrq = [phonemes.probs[j] for j in np.where(np.array(phonemes.phons) == str(consonants[i]))[0].tolist()]
        phprb = phfrq.count(1) / len(phfrq)
        p_phon.append(phprb)
    return Phonemes(
        phons=list(set([consonants[i] for i in [i for i in range(len(p_phon)) if p_phon[i] < 0.05]])),
        probs=[]
    )


def export_speech_synthesiser(syllables: Syllables):
    print("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
    syllables_dir = os.path.join(RESULTS_DEFAULT_PATH, "syllables")
    os.makedirs(syllables_dir, exist_ok=True)
    c = [i[0] for i in syllables.sylls]
    v = [i[1] for i in syllables.sylls]
    c = ' '.join(c).replace('ʃ', 'sch').replace('ɡ', 'g').replace('ç', 'ch').replace('ʒ', 'dsch').split()
    v = ' '.join(v).replace('ɛ', 'ä').replace('ø', 'ö').replace('y', 'ü').split()
    t = [c[i] + v[i] for i in range(len(syllables.sylls))]
    for syllable, text in zip(syllables.sylls, t):
        synth_string = '<phoneme alphabet="ipa" ph=' + '"' + syllable + '"' + '>' + text + '</phoneme>'
        with open(os.path.join(syllables_dir, f'{str(syllable[0:2])}.txt'), 'w') as f:
            f.write(synth_string + "\n")
            csv.writer(f)


def print_obj(obj, n_elements=2):
    kwargs = {}
    s = f"{obj.__class__.__name__}("
    for key, val in obj._asdict().items():
        kwargs[key] = val[:n_elements]
        s += f"{key}=["
        for v in val[:n_elements]:
            if isinstance(v, str):
                s += f'"{v}", '
            elif isinstance(v, list):
                s += f"["
                for vx in v[:n_elements]:
                    s += f'{vx}, '
                s += "...], "
            else:
                s += f'{v}, '
        s += "...], "
    s += ")"
    print(s)


def generate_syllables():
    syllables = read_syllables(SYLLABLES_DEFAULT_PATH)
    print_obj(syllables)

    bin_feats_obj = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)
    print_obj(bin_feats_obj)

    phonemes = read_order_of_phonemes(IPA_SEG_DEFAULT_PATH)
    print_obj(phonemes)

    bigrams = read_bigrams(IPA_BIGRAMS_DEFAULT_PATH)
    trigrams = read_trigrams(IPA_TRIGRAMS_DEFAULT_PATH)
    print_obj(bigrams)
    print_obj(trigrams)

    syllables = select_consonant_vowel(syllables, bin_feats_obj)
    print_obj(syllables)
    syllables = select_occurrence_probability(syllables)
    print_obj(syllables)
    syllables = select_native_phonemes(syllables, phonemes)
    print_obj(syllables)

    print("SELECT BIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    bigrams_uniform = grams_filter_uniform(bigrams)
    print_obj(bigrams_uniform)

    print("SELECT TRIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    trigrams_uniform = grams_filter_uniform(trigrams)
    print_obj(trigrams_uniform)

    rare_onset_phonemes = filter_rare_onset_phonemes(syllables, phonemes)
    print_obj(rare_onset_phonemes)

    print("SAVE SYLLABLES")
    syllables_data = SyllablesData(
        sylls=syllables.sylls,
        phons_rare=rare_onset_phonemes,
        bigrams=bigrams.grams,
        trigrams=trigrams.grams,
        bigrams_prob_filtered=bigrams_uniform.grams,
        trigrams_prob_filtered=trigrams_uniform.grams
    )

    with open(os.path.join(RESULTS_DEFAULT_PATH, "syllables.pickle"), 'wb') as f:
        pickle.dump(syllables_data, f, pickle.HIGHEST_PROTOCOL)

    export_speech_synthesiser(syllables)


if __name__ == '__main__':
    generate_syllables()
