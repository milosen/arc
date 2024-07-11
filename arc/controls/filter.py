from functools import partial
import logging
import os.path
from copy import copy
from os import PathLike
from typing import Iterable, Dict, Union, Optional

import numpy as np
from scipy import stats

from arc.types.base_types import Register, RegisterType
from arc.types.phoneme import Phoneme
from arc.types.syllable import Syllable
from arc.types.word import Word

from arc.io import read_phoneme_corpus, read_bigrams, read_trigrams, IPA_SEG_DEFAULT_PATH, IPA_BIGRAMS_DEFAULT_PATH, \
    IPA_TRIGRAMS_DEFAULT_PATH


logger = logging.getLogger(__name__)


def filter_uniform_syllables(syllables: Register[str, Syllable], alpha: float = 0.05):
    logger.info("Filter uniformly distributed syllables.")
    freqs = [s.info["freq"] for s in syllables]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    return Register({k: v for i, (k, v) in enumerate(syllables.items()) if p_vals_uniform[i] > alpha},
                    _info=copy(syllables.info))


def filter_common_phoneme_syllables(syllables, ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logger.info("Filter syllables with common/native phonemes.")
    native_phonemes = read_phoneme_corpus(ipa_seg_path=ipa_seg_path)

    def is_native(syllable):
        return all((phoneme.id in native_phonemes) for phoneme in syllable)

    return Register({
        syllable.id: syllable for syllable in syllables if is_native(syllable)
    }, _info=copy(syllables.info))


def phoneme_is_common_at(phoneme: Phoneme, position: int = 0, p_threshold: float = 0.05):
    assert "word_position_prob" in phoneme.info.keys(), (
        "To check for phoneme position probability, your phonemes need the 'word_position_prob' info key. " 
        "Before creating syllables and words from your phonemes run `phonemes = phonemes.intersection(read_phoneme_corpus())`."
    )

    return phoneme.info["word_position_prob"].get(position, 0) >= p_threshold


def filter_common_phonemes_at_all_positions(word: Word, p_threshold: float = 0.05) -> bool:
    phonemes = [phoneme for syllable in word for phoneme in syllable]
    return all(phoneme_is_common_at(ph, position, p_threshold=p_threshold) for position, ph in enumerate(phonemes))

def filter_common_phonemes_at_position(word: Word, position, p_threshold: float = 0.05) -> bool:
    phonemes = [phoneme for syllable in word for phoneme in syllable]
    return phoneme_is_common_at(phonemes[position], position, p_threshold=p_threshold)


def filter_common_phoneme_words(words: RegisterType, position: Optional[int] = None, p_threshold: float = 0.05, 
                                ipa_seg_path: Union[str, PathLike] = IPA_SEG_DEFAULT_PATH):
    logger.info("Exclude words with low (onset) syllable probability.")

    if position is None:
        return words.filter(filter_common_phonemes_at_all_positions, p_threshold=p_threshold)
    else:
        return words.filter(filter_common_phonemes_at_position, position=position, p_threshold=p_threshold)


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


def filter_bigrams(words: RegisterType,
                   bigrams_path: Optional[Union[str, PathLike]] = IPA_BIGRAMS_DEFAULT_PATH,
                   p_val: float = None) -> Register[str, Word]:
    logger.info("Select words with uniform bigram and non-zero trigram log-probability of occurrence in the corpus.")

    words = copy(words)  # ?

    words.info.update({"bigram_pval": p_val})

    assert os.path.exists(bigrams_path), "Bigram control requires valid path to bigrams file"

    bigrams = read_bigrams(bigrams_path)

    if p_val is not None:
        bigrams = bigrams.filter(lambda bigram: bigram.info["p_unif"] > p_val)

    words.filter(check_bigram_stats, valid_bigrams=bigrams)

    return words


def filter_trigrams(words: RegisterType,
                    trigrams_path: Optional[Union[str, PathLike]] = IPA_TRIGRAMS_DEFAULT_PATH,
                    p_val: float = None) -> Register[str, Word]:
    logger.info("Select words with uniform bigram and non-zero trigram log-probability of occurrence in the corpus.")

    words = copy(words)  # ?

    words.info.update({"trigram_pval": p_val})

    assert os.path.exists(trigrams_path), "Trigram control requires valid path to trigrams file"
    trigrams = read_trigrams(trigrams_path)

    if p_val is not None:
        trigrams = trigrams.filter(lambda trigram: trigram.info["p_unif"] > p_val)
        
    words.filter(check_trigram_stats, valid_trigrams=trigrams)

    return words
