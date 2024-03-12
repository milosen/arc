import logging
from copy import copy
import random
from typing import Optional, Literal, List, Iterable

from tqdm.rich import tqdm

from arc.controls.stream import StreamType
from arc.controls.lexicon import LexiconType, make_lexicon_generator, sample_random_lexicon
from arc.controls.filter import filter_common_phoneme_words, filter_gram_stats, filter_uniform_syllables, \
    filter_common_phoneme_syllables
from arc.controls.stream import make_stream_from_lexicon
from arc.core.base_types import Register, RegisterType
from arc.core.syllable import make_feature_syllables
from arc.core.word import generate_subset_syllables, Word
from arc.io import read_syllables_corpus


def make_syllables(phonemes: RegisterType, phoneme_pattern: str = "cV",
                   unigram_control: bool = True, unigram_alpha: float = 0.05) -> RegisterType:

    syllables = make_feature_syllables(phonemes, phoneme_pattern=phoneme_pattern)

    if unigram_control:
        german_syllable_corpus = read_syllables_corpus()
        syllables_valid_german = syllables.intersection(german_syllable_corpus)
        syllables_german_filtered = filter_uniform_syllables(syllables_valid_german, alpha=unigram_alpha)
        syllables = filter_common_phoneme_syllables(syllables_german_filtered)

    return syllables


def make_words(syllables: RegisterType,
               num_syllables=3,
               bigram_control=True,
               bigram_alpha=None,
               trigram_control=True,
               trigram_alpha=None,
               positional_control=True,
               phonotactic_control=True,
               n_look_back=2,
               n_words=10_000,
               max_tries=100_000,
               progress_bar: bool = True,
               ) -> RegisterType:
    words = {}

    iter_tries = range(max_tries)

    if progress_bar:
        pbar = tqdm(total=n_words)

    for _ in iter_tries:
        sylls = []
        for _ in range(num_syllables):
            sub = list(filter(lambda x: x.id not in sylls, syllables))
            if phonotactic_control:
                sub = generate_subset_syllables(sub, sylls[-n_look_back:])
            if sub:
                new_rand_valid_syllable = random.sample(sub, 1)[0]
                sylls.append(new_rand_valid_syllable)

        if len(sylls) == num_syllables:
            word_id = "".join(s.id for s in sylls)
            if word_id not in words:
                word_features = list(list(tup) for tup in zip(*[s.info["binary_features"] for s in sylls]))
                words[word_id] = Word(id=word_id, info={"binary_features": word_features}, syllables=sylls)
                if progress_bar:
                    pbar.update(1)

        if len(words) == n_words:
            logging.info(f"Done: Found {n_words} words.")
            break

    words_register = Register(words)
    words_register.info = copy(syllables.info)

    if positional_control:
        words_register = filter_common_phoneme_words(words_register, position=0)

    words_register = filter_gram_stats(
        words_register,
        bigram_control=bigram_control,
        trigram_control=trigram_control,
        p_val_uniform_bigrams=bigram_alpha,
        p_val_uniform_trigrams=trigram_alpha
    )

    return words_register


def make_lexicons(
    words: RegisterType,
    n_lexicons: int = 5,
    n_words: int = 4,
    max_overlap: int = 1,
    lag_of_interest: int = 1,
    max_word_matrix: int = 200,
    unique_words: bool = False,
    control_features: bool = True
) -> List[LexiconType]:

    lexicons = []

    if control_features:
        lexicon_generator = make_lexicon_generator(
            words.get_subset(max_word_matrix),
            n_words=n_words,
            max_overlap=max_overlap,
            lag_of_interest=lag_of_interest
        )
    else:
        lexicon_generator: Iterable = sample_random_lexicon(
            words.get_subset(max_word_matrix),
            n_words=n_words,
        )

    for lexicon in lexicon_generator:

        has_repeating_words = False
        if unique_words:
            # check uniqueness of words across all lexicons
            for lexicon_known in lexicons:
                if set(lexicon_known.keys()).intersection(set(lexicon.keys())):
                    has_repeating_words = True
                    break

        if not has_repeating_words:
            lexicons.append(lexicon)

        if len(lexicons) >= n_lexicons:
            break

    return lexicons


def make_streams(
        lexicons: List[LexiconType],
        max_rhythmicity: Optional[float] = None,
        num_repetitions: int = 4,
        max_tries_randomize: int = 10,
        n_lexicon_streams: int = 1,
        tp_modes: tuple = ("random", "word_structured", "position_controlled")
) -> RegisterType:
    logging.info("Building streams from lexicons ...")

    streams = {}
    n_lex_streams = 0

    for i, lexicon in enumerate(lexicons):
        found_all = True
        new_streams = {}
        for tp_mode in tp_modes:
            stream_id = f"Lexicon-{'-'.join(w.id for w in lexicon)}_TP-{tp_mode}"

            maybe_stream: Optional[StreamType] = make_stream_from_lexicon(
                lexicon,
                max_rhythmicity=max_rhythmicity,
                max_tries_randomize=max_tries_randomize,
                num_repetitions=num_repetitions,
                tp_mode=tp_mode
            )

            if maybe_stream:
                new_streams[stream_id] = maybe_stream
            else:
                found_all = False
                break

        if found_all:
            streams.update(new_streams)
            n_lex_streams += 1

        if n_lex_streams >= n_lexicon_streams:
            break

    streams_register = Register(**streams)

    streams_register.info = {
        "tp_modes": tp_modes,
        "max_rhythmicity": max_rhythmicity,
        "num_repetitions": num_repetitions
    }

    return streams_register
