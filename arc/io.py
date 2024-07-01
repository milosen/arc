import csv
import json
import logging
import os
import pathlib
from importlib import resources as importlib_resources
from os import PathLike
from typing import Iterable, Dict, Union, List, Type, Optional, Literal

import numpy as np
from scipy import stats

from arc.phonecodes import phonecodes
from arc.types.base_types import Register, RegisterType
from arc.types.syllable import Syllable
from arc.types.word import Word
from arc.types.phoneme import PHONEME_FEATURE_LABELS, Phoneme
from arc.types.lexicon import LexiconType

logger = logging.getLogger(__name__)


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


def export_speech_synthesiser(syllables: Iterable[Syllable],
                              syllables_dir: Union[str, PathLike] = SSML_RESULTS_DEFAULT_PATH):
    logger.info("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
    os.makedirs(syllables_dir, exist_ok=True)
    c = [s.id[0] for s in syllables]
    v = [s.id[1] for s in syllables]
    c = ' '.join(c).replace('ʃ', 'sch').replace('ɡ', 'g').replace('ç', 'ch').replace('ʒ', 'dsch').split()
    v = ' '.join(v).replace('ɛ', 'ä').replace('ø', 'ö').replace('y', 'ü').split()
    t = [co + vo for co, vo in zip(c, v)]
    for syllable, text in zip(syllables, t):
        synth_string = '<phoneme alphabet="ipa" ph=' + '"' + syllable.id + '"' + '>' + text + '</phoneme>'
        with open(os.path.join(syllables_dir, f'{str(syllable.id[0:2])}.txt'), 'w') as f:
            f.write(synth_string + "\n")
            csv.writer(f)

    print("Done")


def read_phoneme_corpus(
        ipa_seg_path: Union[os.PathLike, str] = IPA_SEG_DEFAULT_PATH
) -> Register[str, Phoneme]:
    """
    Read order of phonemes, i.e. phonemes from a corpus together with the positions at which they
    appear in a bag of words.

    :param ipa_seg_path:
    :return:
    """
    logger.info("READ ORDER OF PHONEMES IN WORDS")
    with open(ipa_seg_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))
    phonemes = {}
    for phon_data in fdata[1:]:
        phon_data_split = phon_data[0].split(",")
        if len(phon_data_split) == 3:
            phon = phon_data_split[1].replace('"', '').replace("g", "ɡ")
            position_in_word = int(phon_data_split[2])
            if phon in phonemes:
                phonemes[phon].info["order"].append(position_in_word)
            else:
                phonemes[phon] = Phoneme(id=phon, info={"order": [position_in_word]})

    return Register(phonemes)


def syll_to_ipa(syll, language="deu", from_format="xsampa"):
    if from_format == "xsampa":
        return phonecodes.xsampa2ipa(syll, language)
    elif from_format == "ipa":
        return syll
    else:
        raise ValueError(f"Unknown format {from_format}")


def read_syllables_corpus(
        syllables_corpus_path: Union[os.PathLike, str] = SYLLABLES_DEFAULT_PATH,
        from_format: Literal["ipa", "xsampa"] = "xsampa",
        lang: str = "deu",
) -> Register[str, Syllable]:
    logger.info("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")

    with open(syllables_corpus_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    syllables_dict: Dict[str, Syllable] = {}

    for syll_stats in fdata[1:]:
        syll_ipa = syll_to_ipa(syll_stats[1])
        info = {"freq": int(syll_stats[2]), "prob": float(syll_stats[3])}
        if syll_ipa not in syllables_dict or syllables_dict[syll_ipa].info != info:
            syllables_dict[syll_ipa] = Syllable(
                id=syll_ipa, phonemes=[], info=info, binary_features=[], phonotactic_features=[])
        else:
            logger.info(
                f"Syllable '{syll_ipa}' with conflicting stats {info} != {syllables_dict[syll_ipa].info}."
            )
            # del syllables_dict[syll_ipa]

    return Register(syllables_dict)


def read_bigrams(
    ipa_bigrams_path: str = IPA_BIGRAMS_DEFAULT_PATH,
) -> Register[str, Syllable]:
    logger.info("READ BIGRAMS")

    with open(ipa_bigrams_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    freqs = [int(data[0].split(",")[2]) for data in fdata[1:]]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    bigrams_dict: Dict[str, Syllable] = {}

    for bigram_stats, p_unif in zip(fdata[1:], p_vals_uniform):
        bigram_stats = bigram_stats[0].split(",")
        bigram = bigram_stats[1].replace('_', '').replace("g", "ɡ")
        info = {"freq": int(bigram_stats[2]), "p_unif": p_unif}

        if bigram not in bigrams_dict or bigrams_dict[bigram].info == info:
            # a bigram is not necessarily a syllable but in our type system they are equivalent
            bigrams_dict[bigram] = Syllable(id=bigram, phonemes=[], info=info)
        else:
            logger.info(
                f"Bigram '{bigram}' with conflicting stats {info} != {bigrams_dict[bigram].info}."
            )
            # del bigrams_dict[bigram]

    return Register(bigrams_dict)


def read_trigrams(
        ipa_trigrams_path: str = IPA_TRIGRAMS_DEFAULT_PATH,
) -> Register[str, Syllable]:
    logger.info("READ TRIGRAMS")
    fdata = list(csv.reader(open(ipa_trigrams_path, "r"), delimiter='\t'))

    freqs = [int(data[0].split(",")[1]) for data in fdata[1:]]
    p_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    trigrams_dict: Dict[str, Syllable] = {}
    for data, p_unif in zip(fdata[1:], p_uniform):
        data = data[0].split(",")
        trigram = data[0].replace('_', '').replace("g", "ɡ")
        info = {"freq": int(data[1]), "p_unif": p_unif}

        if trigram not in trigrams_dict or trigrams_dict[trigram].info == info:
            trigrams_dict[trigram] = Syllable(id=trigram, phonemes=[], info=info)
        else:
            logger.info(
                f"Trigram '{trigram}' with conflicting stats {info} != {trigrams_dict[trigram].info}."
            )
            # del trigrams_dict[trigram]

    return Register(trigrams_dict)


def read_phonemes_csv(binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH) -> Register:
    logger.info("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")

    with open(binary_features_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file))

    phons = [row[0] for row in fdata[1:]]
    feats = [row[1:] for row in fdata[1:]]
    phoneme_feature_labels = fdata[0][1:]

    assert phoneme_feature_labels == PHONEME_FEATURE_LABELS

    phonemes_dict = {}
    for phon, features in zip(phons, feats):
        if phon not in phonemes_dict or features == phonemes_dict[phon].info["features"]:
            phonemes_dict[phon] = Phoneme(id=phon, info={"features": features})
        else:
            logger.info(
                f"Phoneme '{phon}' with conflicting "
                f"feature entries {features} != {phonemes_dict[phon].info['features']}.")
            # del phonemes_dict[phon]

    return Register(phonemes_dict, _info={"phoneme_feature_labels": phoneme_feature_labels})


def check_german(words: List[Word]):
    # TODO
    # SAVE WORDS IN ONE CSV FILE
    with open(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv'), 'w') as f:
        writer = csv.writer(f)
        for word in words:
            writer.writerows([[word.id, 0]])

    # TO ENSURE THAT THE TRIPLETS CANNOT BE MISTAKEN FOR GERMAN WORDS,
    # WE INSTRUCTED A NATIVE GERMAN SPEAKER TO MARK EACH TRIPLET AS...
    #     '1' IF IT CORRESPONDS EXACTLY TO A GERMAN WORD
    #     '2' IF IT COULD BE MISTAKEN FOR A GERMAN WORD-GROUP WHEN PRONOUNCED ALOUD
    #     '3' IF THE PRONUNCIATION OF THE FIRST TWO SYLLABLES IS A WORD CANDIDATE,
    #         i.e. the syllable pair could be mistaken for a German word, or
    #         it evokes a strong prediction for a certain real German word
    #     '4' IF IT DOES NOT SOUND GERMAN AT ALL
    #         (that is, if the phoneme combination is illegal in German morphology
    #         [do not flag if rule exceptions exist])
    #     '0' OTHERWISE (that is, the item is good)

    logger.info("LOAD WORDS FROM CSV FILE AND SELECT THOSE THAT CANNOT BE MISTAKEN FOR GERMAN WORDS")
    with open(os.path.join(RESULTS_DEFAULT_PATH, "words.csv"), 'r') as f:
        fdata = list(csv.reader(f, delimiter='\t'))
    rows = [row[0].split(",") for row in fdata]
    words = [row[0] for row in rows if row[1] == "0"]

    return words


def load_default_phonemes():
    return read_phonemes_csv()


def arc_register_from_json(path: Union[str, PathLike], arc_type: Type) -> RegisterType:
    """
    Load an arc register from a json file.
    """
    with open(path, "r") as file:
        d = json.load(file)

    # we have to process the "_info" field separately because it's not a valid ARC type
    register = Register({k: arc_type(**v) for k, v in d.items() if k != "_info"})
    register.info = d["_info"]

    return register


def load_phonemes(path_to_json: Optional[Union[str, PathLike]] = None) -> RegisterType:
    if path_to_json is None:
        return load_default_phonemes()

    return arc_register_from_json(path_to_json, Phoneme)


def load_syllables(path_to_json: Union[str, PathLike]):
    return arc_register_from_json(path_to_json, Syllable)


def load_words(path_to_json: Union[str, PathLike]):
    return arc_register_from_json(path_to_json, Word)


def load_lexicons(path_to_json: Union[str, PathLike]):
    return arc_register_from_json(path_to_json, LexiconType)
