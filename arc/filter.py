import logging
from copy import copy
from os import PathLike
from typing import Iterable, Dict, List, Union, Optional

import numpy as np
from scipy import stats
from tqdm.rich import tqdm

from arc.io import read_phoneme_corpus, read_bigrams, read_trigrams, IPA_SEG_DEFAULT_PATH, IPA_BIGRAMS_DEFAULT_PATH, \
    IPA_TRIGRAMS_DEFAULT_PATH
from arc.types import Syllable, Phoneme, Word, Register


def filter_uniform_syllables(syllables: Register[str, Syllable]):
    logging.info("FILTER UNIFORMLY DISTRIBUTED SYLLABLES")
    freqs = [s.info["freq"] for s in syllables]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    return Register({k: v for i, (k, v) in enumerate(syllables.items()) if p_vals_uniform[i] > 0.05},
                    _info=copy(syllables.info))


def filter_common_phoneme_syllables(syllables, ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logging.info("FILTER SYLLABLES WITH COMMON/NATIVE PHONEMES")
    native_phonemes = read_phoneme_corpus(ipa_seg_path=ipa_seg_path)

    def is_native(syllable):
        return all((phoneme.id in native_phonemes) for phoneme in syllable)

    return Register({
        syllable.id: syllable for syllable in syllables if is_native(syllable)
    }, _info=copy(syllables.info))


def get_rare_phonemes(syllables: Iterable[Syllable], phonemes: Dict[str, Phoneme],
                      position: int = 0, p_threshold: float = 0.05):
    """
    Get phonemes that are rarely (in the sense that p_val < 0.05) found at `position` of a word
    :param syllables:
    :param phonemes:
    :param position:
    :param p_threshold:
    :return:
    """
    logging.info("FIND SYLLABLES THAT ARE RARE AT THE ONSET OF A WORD")

    rare_onset_phonemes = []
    for s in syllables:
        phon = s.id[position]
        if phon in phonemes:
            phoneme_prob = phonemes[phon].info["order"].count(position + 1) / len(phonemes[phon].info["order"])
        else:
            phoneme_prob = 0
        if phoneme_prob < p_threshold:
            rare_onset_phonemes.append(s[position])

    return rare_onset_phonemes


def filter_common_phoneme_words(words, position: int = 0, ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logging.info("EXCLUDE WORDS WITH LOW (ONSET) SYLLABLE PROBABILITY")
    native_phonemes = read_phoneme_corpus(ipa_seg_path=ipa_seg_path)
    list_syllables = [syllable for word in words for syllable in word]

    rare_phonemes = get_rare_phonemes(list_syllables, native_phonemes)

    # e.g. word[syll_idx=0][phon_idx=0] means first phoneme in first syllable of the word
    if position == -1:
        syll_idx, phon_idx = -1, -1
    else:
        phons_in_syll = len(words[0][0].phonemes)
        syll_idx, phon_idx = position // phons_in_syll, position % phons_in_syll

    return Register({
        word.id: word for word in tqdm(words) if word[syll_idx][phon_idx] not in rare_phonemes
    }, _info=copy(words.info))


def check_bigram_stats(word: Word, valid_bigrams: Register[str, Syllable]):
    phonemes = [phon for syll in word.syllables for phon in syll.phonemes]

    for phon_1, phon_2 in zip(phonemes[:-1], phonemes[1:]):
        if "".join([phon_1.id, phon_2.id]) not in valid_bigrams:
            return False

    return True


def check_trigram_stats(word: Word, valid_trigrams: Register[str, Syllable]):
    phonemes = [phon for syll in word.syllables for phon in syll.phonemes]

    for phon_1, phon_2, phon_3 in zip(phonemes[:-2], phonemes[1:-1], phonemes[2:]):
        if "".join([phon_1.id, phon_2.id, phon_3.id]) not in valid_trigrams:
            return False

    return True


def filter_gram_stats(words: Register[str, Word],
                      bigrams: Optional[Union[str, PathLike]] = IPA_BIGRAMS_DEFAULT_PATH,
                      trigrams: Optional[Union[str, PathLike]] = IPA_TRIGRAMS_DEFAULT_PATH,
                      uniform_bigrams=False, uniform_trigrams=False) -> Register[str, Word]:
    logging.info("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")

    if not bigrams and not trigrams:
        logging.info("Nothing to do. Please supply bigrams and/or trigrams path")
        return words

    info = copy(words.info)

    filtered_words = Register(**words)
    if bigrams:
        bigrams: Register[str, Syllable] = read_bigrams(bigrams)
        if uniform_bigrams:
            bigrams = Register({
                k: v for bigram in bigrams for k, v in bigram.items() if bigram.info["p_unif"] > 0.05
            })
            info.update({"bigram_pval": 0.05})
        info.update({"bigrams": bigrams, "bigrams_count": len(bigrams)})
        filtered_words_dict = {word.id: word for word in filtered_words if check_bigram_stats(word, bigrams)}
        filtered_words = Register(**filtered_words_dict)

    if trigrams:
        trigrams: Register[str, Syllable] = read_trigrams(trigrams)
        if uniform_trigrams:
            trigrams_dict = {k: v for bigram in bigrams for k, v in bigram.items() if bigram.info["p_unif"] > 0.05}
            trigrams = Register(**trigrams_dict)
            info.update({"trigram_pval": 0.05})
        info.update({"trigrams": trigrams, "trigrams_count": len(trigrams)})
        filtered_words_dict = {word.id: word for word in filtered_words if check_trigram_stats(word, trigrams)}
        filtered_words = Register(**filtered_words_dict)

    filtered_words.info = info

    return filtered_words
