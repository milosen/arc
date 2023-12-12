import logging
from os import PathLike
from typing import Iterable, Dict, List, Union, Optional

import numpy as np
from scipy import stats

from arc.definitions import *
from arc.functional import add_custom_features
from arc.io import read_syllables_corpus, read_ipa_seg_order_of_phonemes, read_bigrams, read_trigrams
from arc.types import Syllable, Phoneme, Word, CollectionARC


def filter_uniform_syllables(syllables: CollectionARC[str, Syllable]):
    logging.info("FILTER UNIFORMLY DISTRIBUTED SYLLABLES")
    freqs = [s.info["freq"] for s in syllables]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    return CollectionARC({k: v for i, (k, v) in enumerate(syllables.items()) if p_vals_uniform[i] > 0.05})


def filter_with_corpus(
        syllables: CollectionARC[str, Syllable],
        syllable_corpus: Union[PathLike, str, CollectionARC[str, Syllable]] = SYLLABLES_DEFAULT_PATH
) -> CollectionARC[str, Syllable]:
    """
    Select syllables that are also in the corpus and merge the data
    :param syllables:
    :param syllable_corpus:
    :return:
    """
    if isinstance(syllable_corpus, (PathLike, str)):
        syllable_corpus = read_syllables_corpus(syllable_corpus)

    intersection = CollectionARC()

    for key, corpus_syllable in syllable_corpus.items():
        if key in syllables:
            intersection[key] = syllables[key]
            intersection[key].info = corpus_syllable.info

    return intersection


def filter_common_phoneme_syllables(syllables, ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logging.info("FILTER SYLLABLES WITH COMMON/NATIVE PHONEMES")
    native_phonemes = read_ipa_seg_order_of_phonemes(ipa_seg_path=ipa_seg_path)

    return CollectionARC({
        syllable.id: syllable for syllable in syllables if all((phoneme.id in native_phonemes) for phoneme in syllable)
    })


def get_rare_onset_phonemes(syllables: Iterable[Syllable], phonemes: Dict[str, Phoneme],
                            p_threshold: float = 0.05):
    logging.info("FIND SYLLABLES THAT ARE RARE AT THE ONSET OF A WORD")

    rare_onset_phonemes = []
    for s in syllables:
        phon = s.id[0]
        if phon in phonemes:
            phoneme_prob = phonemes[phon].order.count(1) / len(phonemes[phon].order)
        else:
            phoneme_prob = 0
        if phoneme_prob < p_threshold:
            rare_onset_phonemes.append(s[0])

    return rare_onset_phonemes


def filter_common_onset_words(words, ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logging.info("EXCLUDE WORDS WITH LOW ONSET SYLLABLE PROBABILITY")
    native_phonemes = read_ipa_seg_order_of_phonemes(ipa_seg_path=ipa_seg_path)
    list_syllables = [syllable for word in words for syllable in word]

    rare_phonemes = get_rare_onset_phonemes(list_syllables, native_phonemes)
    logging.info("Rare onset phonemes:", [p.id for p in rare_phonemes])

    # word[0][0] = first phoneme in first syllable of the word
    return CollectionARC({
        word.id: word for word in words if word[0][0] not in rare_phonemes
    })


def check_bigram_stats(word: Word, valid_bigrams: CollectionARC[str, Syllable]):
    phonemes = [phon for syll in word.syllables for phon in syll.phonemes]

    for phon_1, phon_2 in zip(phonemes[:-1], phonemes[1:]):
        if "".join([phon_1.id, phon_2.id]) not in valid_bigrams:
            return False

    return True


def check_trigram_stats(word: Word, valid_trigrams: CollectionARC[str, Syllable]):
    phonemes = [phon for syll in word.syllables for phon in syll.phonemes]

    for phon_1, phon_2, phon_3 in zip(phonemes[:-2], phonemes[1:-1], phonemes[2:]):
        if "".join([phon_1.id, phon_2.id, phon_3.id]) not in valid_trigrams:
            return False

    return True


def filter_gram_stats(words: CollectionARC[str, Word],
                      bigrams: Optional[Union[str, PathLike]] = IPA_BIGRAMS_DEFAULT_PATH,
                      trigrams: Optional[Union[str, PathLike]] = IPA_TRIGRAMS_DEFAULT_PATH,
                      uniform_bigrams=False, uniform_trigrams=False) -> CollectionARC[str, Word]:
    logging.info("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")

    if not bigrams and not trigrams:
        logging.info("Nothing to do. Please supply bigrams and/or trigrams path")
        return words

    filtered_words = CollectionARC(**words)
    if bigrams:
        bigrams: CollectionARC[str, Syllable] = read_bigrams(bigrams)
        if uniform_bigrams:
            bigrams = CollectionARC({
                k: v for bigram in bigrams for k, v in bigram.items() if bigram.info["p_unif"] > 0.05
            })
        filtered_words_dict = {word.id: word for word in filtered_words if check_bigram_stats(word, bigrams)}
        filtered_words = CollectionARC(**filtered_words_dict)

    if trigrams:
        trigrams: CollectionARC[str, Syllable] = read_trigrams(trigrams)
        if uniform_trigrams:
            trigrams_dict = {k: v for bigram in bigrams for k, v in bigram.items() if bigram.info["p_unif"] > 0.05}
            trigrams = CollectionARC(**trigrams_dict)
        filtered_words_dict = {word.id: word for word in filtered_words if check_trigram_stats(word, trigrams)}
        filtered_words = CollectionARC(**filtered_words_dict)

    return filtered_words
