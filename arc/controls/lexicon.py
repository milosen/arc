import itertools
import logging
import math
from copy import copy
from functools import partial
from typing import Generator, Tuple, Set

import numpy as np

from arc.core.word import word_overlap_matrix, WordType
from arc.core.syllable import SyllableType
from arc.core.base_types import Register, RegisterType


LexiconType = RegisterType


def word_as_syllable_set(word: WordType) -> Set[SyllableType]:
    return set(syllable.id for syllable in word)


def check_no_syllable_overlap(words, word_index_pair):
    index_word_1, index_word_2 = set(word_index_pair)

    syllables_word_1 = word_as_syllable_set(words[index_word_1])
    syllables_word_2 = word_as_syllable_set(words[index_word_2])

    intersection = syllables_word_1.intersection(syllables_word_2)

    return not intersection


def make_lexicon_generator(
        words: RegisterType,
        n_words: int = 4,
        max_overlap: int = 1,
        max_yields: int = 10) -> Generator[LexiconType, None, None]:

    overlap = word_overlap_matrix(words)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    logging.info(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0

    iter_allowed_overlaps = itertools.product(range(max_overlap + 1), range(1, math.comb(n_words, 2)))

    for max_pair_overlap, max_overlap_with_n_words in iter_allowed_overlaps:

        max_cum_overlap = max_pair_overlap * max_overlap_with_n_words

        if max_pair_overlap != 0:
            logging.warning(
                f"Increasing allowed overlaps: "
                f"MAX_PAIRWISE_OVERLAP={max_pair_overlap}, "
                f"MAX_CUMULATIVE_OVERLAP={max_cum_overlap}"
            )

        # WORDSxWORDS boolean matrix indicating if the words can be paired together based on the maximum overlap
        valid_word_pairs_matrix = (overlap <= max_pair_overlap)

        # represent the matrix from above as a set of pairs of word indexes, e.g. {{0, 1}, {0, 2}, ...}
        min_overlap_pairs = zip(*np.where(valid_word_pairs_matrix))
        min_overlap_pairs = set(frozenset([int(pair[0]), int(pair[1])]) for pair in min_overlap_pairs)

        # select only those pairs that have pairwise unique syllables
        no_syllable_overlap = partial(check_no_syllable_overlap, words)
        min_overlap_pairs = set(filter(no_syllable_overlap, min_overlap_pairs))

        for start_pair in min_overlap_pairs:

            lexicon_indexes = set(start_pair)
            cumulative_overlap = 0

            for candidate_idx in range(len(overlap)):

                # is candidate word known?
                if candidate_idx in lexicon_indexes:
                    continue

                # does the new word exceed the allowed pairwise overlap?
                has_min_overlap = [({known_idx, candidate_idx} in min_overlap_pairs) for known_idx in lexicon_indexes]
                if not all(has_min_overlap):
                    continue

                # make sure the new word does not increase the cumulative overlap too much
                sum_overlaps_with_known = sum([overlap[known_idx, candidate_idx] for known_idx in lexicon_indexes])
                current_overlap_budget = (max_cum_overlap - cumulative_overlap)
                if sum_overlaps_with_known > current_overlap_budget:
                    continue

                # success! let's add the new word
                lexicon_indexes.add(candidate_idx)
                cumulative_overlap += sum_overlaps_with_known

                # do we have to go again?
                if len(lexicon_indexes) < n_words:
                    continue

                # yield lexicon (guaranteed to be the next best)
                lexicon = Register({words[idx].id:  words[idx] for idx in lexicon_indexes})
                lexicon.info = copy(words.info)
                lexicon.info["cumulative_feature_repetitiveness"] = cumulative_overlap
                lexicon.info["max_pairwise_feature_repetitiveness"] = max_pair_overlap
                yield lexicon

                yields += 1

                if yields == max_yields:
                    return


def make_lexicons_from_words(
        words: RegisterType, n_lexicons: int = 5, n_words: int = 4, max_overlap: int = 1,
) -> Tuple[LexiconType, ...]:
    lexicon_generator = make_lexicon_generator(words, n_words=n_words, max_yields=n_lexicons, max_overlap=max_overlap)
    return tuple(lexicon for lexicon in lexicon_generator)