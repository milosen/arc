import itertools
import logging
import math
from copy import copy
from typing import Generator, Tuple

import numpy as np

from arc.generation.words import word_overlap_matrix
from arc.types import Word, Register, Lexicon, TypeRegister


# TODO: Make pretty from here


def make_lexicon_generator(
        words: Register[str, Word],
        n_words: int = 4,
        max_overlap: int = 1,
        max_yields: int = 10) -> Generator[Lexicon, None, None]:

    overlap = word_overlap_matrix(words)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    logging.info(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0

    def check_no_syllable_overlap(pair):
        w1, w2 = pair
        intersection = set(syllable.id for syllable in words[w1]) & set(syllable.id for syllable in words[w2])
        return not intersection

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
        min_overlap_pairs = set(filter(check_no_syllable_overlap, min_overlap_pairs))

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
                lexicon.info["cumulative_overlap"] = cumulative_overlap
                lexicon.info["max_pairwise_overlap"] = max_pair_overlap
                yield lexicon

                yields += 1

                if yields == max_yields:
                    return


def make_lexicons_from_words(
        words: TypeRegister, n_lexicons: int = 5, n_words: int = 4, max_overlap: int = 1,
) -> Tuple[Lexicon, ...]:
    lexicon_generator = make_lexicon_generator(words, n_words=n_words, max_yields=n_lexicons, max_overlap=max_overlap)
    return tuple(lexicon for lexicon in lexicon_generator)
