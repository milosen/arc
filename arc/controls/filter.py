import logging
import os.path
from copy import copy
from os import PathLike
from typing import Iterable, Dict, Union, Optional

import numpy as np
from scipy import stats
from tqdm import tqdm

from arc.io import read_phoneme_corpus, read_bigrams, read_trigrams, IPA_SEG_DEFAULT_PATH, IPA_BIGRAMS_DEFAULT_PATH, \
    IPA_TRIGRAMS_DEFAULT_PATH
from arc.core.base_types import Register
from arc.core.phoneme import Phoneme
from arc.core.syllable import Syllable
from arc.core.word import Word


def filter_uniform_syllables(syllables: Register[str, Syllable], alpha: float = 0.05):
    logging.info("FILTER UNIFORMLY DISTRIBUTED SYLLABLES")
    freqs = [s.info["freq"] for s in syllables]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    return Register({k: v for i, (k, v) in enumerate(syllables.items()) if p_vals_uniform[i] > alpha},
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


def filter_common_phoneme_words(words, position: Optional[int] = None, ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logging.info("EXCLUDE WORDS WITH LOW (ONSET) SYLLABLE PROBABILITY")
    native_phonemes = read_phoneme_corpus(ipa_seg_path=ipa_seg_path)
    list_syllables = [syllable for word in words for syllable in word]

    rare_phonemes = get_rare_phonemes(list_syllables, native_phonemes)

    if position is None:
        reg = Register({}, _info=copy(words.info))
        for word in words:
            if all(ph not in rare_phonemes for syll in word for ph in syll):
                reg.append(word)
        return reg
    else:
        # e.g. word[syll_idx=0][phon_idx=0] means first phoneme in first syllable of the word
        if position == -1:
            syll_idx, phon_idx = -1, -1
        else:
            phons_in_syll = len(words[0][0].phonemes)
            syll_idx, phon_idx = position // phons_in_syll, position % phons_in_syll

        return Register({
            word.id: word for word in words if word[syll_idx][phon_idx] not in rare_phonemes
        }, _info=copy(words.info))


def check_bigram_stats(word: Word, valid_bigrams: Register[str, Syllable]):
    phonemes = [phon for syllable in word for phon in syllable]

    for phon_1, phon_2 in zip(phonemes[:-1], phonemes[1:]):
        if "".join([phon_1.id, phon_2.id]) not in valid_bigrams:
            return False

    return True


def check_trigram_stats(word: Word, valid_trigrams: Register[str, Syllable]):
    phonemes = [phon for syllable in word for phon in syllable]

    for phon_1, phon_2, phon_3 in zip(phonemes[:-2], phonemes[1:-1], phonemes[2:]):
        if "".join([phon_1.id, phon_2.id, phon_3.id]) not in valid_trigrams:
            return False

    return True


def filter_gram_stats(words: Register[str, Word],
                      bigram_control: bool = True,
                      trigram_control: bool = True,
                      bigrams_path: Optional[Union[str, PathLike]] = IPA_BIGRAMS_DEFAULT_PATH,
                      trigrams_path: Optional[Union[str, PathLike]] = IPA_TRIGRAMS_DEFAULT_PATH,
                      p_val_uniform_bigrams: float = None,
                      p_val_uniform_trigrams: float = None) -> Register[str, Word]:
    logging.info("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")

    info = copy(words.info)

    filtered_words = Register(**words)

    if bigram_control:
        print("bigram control...")
        assert os.path.exists(bigrams_path), "Bigram Control requires valid path to bigrams file"
        bigrams: Register[str, Syllable] = read_bigrams(bigrams_path)
        if p_val_uniform_bigrams is not None:
            bigrams = Register({
                k: bigram for k, bigram in bigrams.items() if bigram.info["p_unif"] > p_val_uniform_bigrams
            })
        info.update({"bigram_pval": p_val_uniform_bigrams, "bigrams_count": len(bigrams)})
        filtered_words_dict = {word.id: word for word in filtered_words if check_bigram_stats(word, bigrams)}
        filtered_words = Register(**filtered_words_dict)

    if trigram_control:
        print("trigram control...")
        assert os.path.exists(trigrams_path), "Trigram Control requires valid path to trigrams file"
        trigrams: Register[str, Syllable] = read_trigrams(trigrams_path)
        if p_val_uniform_trigrams is not None:
            trigrams = Register({
                k: trigram for k, trigram in trigrams.items() if trigram.info["p_unif"] > p_val_uniform_trigrams
            })
        info.update({"trigram_pval": p_val_uniform_trigrams, "trigrams_count": len(trigrams)})
        filtered_words_dict = {word.id: word for word in filtered_words if check_trigram_stats(word, trigrams)}
        filtered_words = Register(**filtered_words_dict)

    filtered_words.info = info

    return filtered_words
