import csv
import logging
import pickle
from os import PathLike
from typing import Iterable, Dict, Union, List

import numpy as np
from scipy import stats
from tqdm.rich import tqdm

from arc.definitions import *
from arc.phonecodes import phonecodes
from arc.types import Syllable, Phoneme, CollectionARC, Word


def export_speech_synthesiser(syllables: Iterable[Syllable],
                              syllables_dir: Union[str, PathLike] = SSML_RESULTS_DEFAULT_PATH):
    logging.info("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
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


def maybe_load_from_file(path, force_redo: bool = False):
    def _outer_wrapper(wrapped_function):
        def _wrapper(*args, **kwargs):
            if not os.path.exists(path) or force_redo:
                logging.info("NO DATA FOUND, RUNNING AGAIN.")
                data = wrapped_function(*args, **kwargs)

                with open(path, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

            else:
                logging.info("SKIPPING GENERATION. LOADING DATA FROM FILE.")
                with open(path, 'rb') as f:
                    data = pickle.load(f)

            return data
        return _wrapper
    return _outer_wrapper


def read_ipa_seg_order_of_phonemes(
        ipa_seg_path: Union[os.PathLike, str] = IPA_SEG_DEFAULT_PATH
) -> CollectionARC[str, Phoneme]:
    """
    Read order of phonemes, i.e. phonemes from a corpus together with the positions at which they
    appear in a bag of words.

    :param ipa_seg_path:
    :return:
    """
    logging.info("READ ORDER OF PHONEMES IN WORDS")
    with open(ipa_seg_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))
    phonemes = {}
    for phon_data in tqdm(fdata[1:]):
        phon_data_split = phon_data[0].split(",")
        if len(phon_data_split) == 3:
            phon = phon_data_split[1].replace('"', '').replace("g", "ɡ")
            position_in_word = int(phon_data_split[2])
            if phon in phonemes:
                phonemes[phon].order.append(position_in_word)
            else:
                phonemes[phon] = Phoneme(id=phon, order=[position_in_word], features=[], info={})

    return CollectionARC(phonemes)


def read_syllables_corpus(
        syllables_corpus_path: Union[os.PathLike, str] = SYLLABLES_DEFAULT_PATH,
) -> CollectionARC[str, Syllable]:
    logging.info("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")

    with open(syllables_corpus_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    syllables_dict: Dict[str, Syllable] = {}

    for syll_stats in fdata[1:]:
        syll_ipa = phonecodes.xsampa2ipa(syll_stats[1], 'deu')
        info = {"freq": int(syll_stats[2]), "prob": float(syll_stats[3])}
        if syll_ipa not in syllables_dict or syllables_dict[syll_ipa].info != info:
            syllables_dict[syll_ipa] = Syllable(
                id=syll_ipa, phonemes=[], info=info, binary_features=[], phonotactic_features=[])
        else:
            logging.warning(
                f"Dropping syllable '{syll_ipa}' with conflicting stats {info} != {syllables_dict[syll_ipa].info}."
            )
            del syllables_dict[syll_ipa]

    return CollectionARC(syllables_dict)


def read_bigrams(
    ipa_bigrams_path: str = IPA_BIGRAMS_DEFAULT_PATH,
) -> CollectionARC[str, Syllable]:
    logging.info("READ BIGRAMS")

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
            bigrams_dict[bigram] = Syllable(
                id=bigram, phonemes=[], info=info, binary_features=[], phonotactic_features=[]
            )
        else:
            logging.warning(
                f"Dropping bigram '{bigram}' with conflicting stats {info} != {bigrams_dict[bigram].info}."
            )
            del bigrams_dict[bigram]

    return CollectionARC(bigrams_dict)


def read_trigrams(
        ipa_trigrams_path: str = IPA_TRIGRAMS_DEFAULT_PATH,
) -> CollectionARC[str, Syllable]:
    logging.info("READ TRIGRAMS")
    fdata = list(csv.reader(open(ipa_trigrams_path, "r"), delimiter='\t'))

    freqs = [int(data[0].split(",")[1]) for data in fdata[1:]]
    p_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    trigrams_dict: Dict[str, Syllable] = {}
    for data, p_unif in zip(fdata[1:], p_uniform):
        data = data[0].split(",")
        trigram = data[0].replace('_', '').replace("g", "ɡ")
        info = {"freq": int(data[1]), "p_unif": p_unif}
        if trigram not in trigrams_dict or trigrams_dict[trigram].info == info:
            trigrams_dict[trigram] = Syllable(
                id=trigram, phonemes=[], info=info, binary_features=[], phonotactic_features=[]
            )
        else:
            logging.warning(
                f"Dropping trigram '{trigram}' with conflicting stats {info} != {trigrams_dict[trigram].info}."
            )
            del trigrams_dict[trigram]

    return CollectionARC(trigrams_dict)


def read_phoneme_features(
        binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH,
        return_as_dict: bool = False,
) -> Union[Iterable[Phoneme], Dict[str, Phoneme]]:
    logging.info("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")

    with open(binary_features_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file))

    phons = [row[0] for row in fdata[1:]]
    feats = [row[1:] for row in fdata[1:]]

    phonemes_dict = {}
    for phon, features in zip(phons, feats):
        if phon not in phonemes_dict or features == phonemes_dict[phon].features:
            phonemes_dict[phon] = Phoneme(id=phon, features=features, order=[], info={})
        else:
            logging.warning(f"Dropping phoneme '{phon}' with conflicting feature entries {features} != {phonemes_dict[phon].features}.")
            del phonemes_dict[phon]

    if return_as_dict:
        return phonemes_dict

    return phonemes_dict.values()


def read_phonemes(binary_features_path: str = PHONEMES_DEFAULT_PATH) -> CollectionARC:
    logging.info("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")

    import json

    with open(binary_features_path, 'r') as f:
        data = json.load(f)

    phonemes_collection = {}
    CollectionARC({key: Phoneme(**values) for key, values in data.items()})
    for key, values in data.items():
        if key not in phonemes_collection or values["binary_features"] == phonemes_collection[key]["binary_features"]:
            phonemes_collection[key] = Phoneme(**values)
        else:
            logging.warning(
                f"Dropping phoneme '{key}' with conflicting feature entries {values['binary_features']} != {phonemes_collection[key]['binary_features']}.")
            del phonemes_collection[key]

    return CollectionARC(phonemes_collection)


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

    logging.info("LOAD WORDS FROM CSV FILE AND SELECT THOSE THAT CANNOT BE MISTAKEN FOR GERMAN WORDS")
    with open(os.path.join(RESULTS_DEFAULT_PATH, "words.csv"), 'r') as f:
        fdata = list(csv.reader(f, delimiter='\t'))
    rows = [row[0].split(",") for row in fdata]
    words = [row[0] for row in rows if row[1] == "0"]

    return words
