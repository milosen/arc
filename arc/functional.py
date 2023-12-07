import collections
import itertools
import logging
import math
import pickle
import random
from functools import reduce
from typing import Callable, Iterable, Union, Generator, Set
import csv

import numpy as np
from scipy import stats
from tqdm.rich import tqdm

from arc.definitions import *
from arc.phonecodes import phonecodes
from arc.types import *


def transitional_p_matrix(v):
    n = 1 + max(v)
    M = [[0] * n for _ in range(n)]
    for (i, j) in zip(v, v[1:]):
        M[i][j] += 1
    for r in M:
        s = sum(r)
        if s > 0:
            r[:] = [i/s for i in r]
    return M


# GENERATE SERIES OF SYLLABLES WITH UNIFORM TPs
def pseudo_walk_tp_random(P, T, v, S, N, n_words=4, n_sylls_per_word=3):
    # TODO
    t = []
    for iTrip in range(n_words):
        for iPoss in range(n_sylls_per_word):
            if not v and not t:
                if N:
                    x = [N]
                    N = []
                else:
                    x = random.sample(P[iPoss], 1)
                t += x
            else:
                iLoop = 0
                while iLoop < 1000:
                    x = random.sample([i for i in P[iPoss] if i not in t], 1)[0]
                    if not t:
                        s = v[-1]
                    else:
                        s = t[-1]
                    if [s, x] not in S and [s, x] in T[iPoss-1]:
                        S.append([s, x])
                        t.append(x)
                        break
                    else:
                        iLoop += 1
    if iLoop < 1000:
        v += t
    return v, S, t


def pseudo_rand_tp_random(n_words=4, n_sylls_per_word=3):
    # TODO
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_repetitions = n_sylls_total * 4  # number of repetitions in a trial
    P = [list(range(i, n_sylls_total, n_sylls_per_word)) for i in range(n_sylls_per_word)]
    I = list(range(n_sylls_per_word))
    B = I[:]
    B.append(B.pop(0))
    T = [[list(i) for i in list(itertools.product(*[P[j], P[k]]))] for j, k in zip(I, B)]
    while True:
        V = []
        while len(V) < n_sylls_total * n_repetitions:
            v = []
            t = []
            if not V:
                N = []
                S = []
            else:
                N = [i for i in T[2] if i not in S][0][1]
                S = []
            while len(v) < len(T) * len(T[0]):
                v, S, t = pseudo_walk_tp_random(P, T, v, S, N)
                if len(t) < n_sylls_total:
                    v = []
                    S = []
                    t = []
            V += v
        V.append(V[0])
        M = np.array(transitional_p_matrix(V))
        V.pop()
        if all(set(i) <= set([1/n_words, 0]) for i in M):
            break
    return V, M


# GENERATE RANDOMIZATIONS CONTROLLING FOR UNIFORMITY OF TRANSITION PROBABILITIES ACROSS WORDS
def pseudo_rand_tp_struct(n_words=4, n_sylls_per_word=3):
    # TODO
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_repetitions = n_sylls_total * 4  # number of repetitions in a trial
    while True:
        P = list(itertools.permutations(np.arange(n_words)))
        S = [[*(i for _ in range(math.ceil(n_repetitions / len(P))))] for i in P]
        M = np.zeros((n_words, n_words))
        v = []
        c = 2
        mask = (np.ones((n_words, n_words)) - np.diag(np.ones(n_words))).astype(bool)
        for _ in range(n_repetitions):
            R = [S[i] for i in range(len(S)) if len(S[i]) == max(map(len, S))]
            if not v:
                p = list(list(random.sample(R, 1)[0])[0])
                m = np.array(transitional_p_matrix(p))
            else:
                s = v[-1]
                n = list(np.where(M[s] == M[s].min())[0])
                n.remove(s)
                if not n:
                    n = list(np.where(M[s] == np.sort(M[s])[1])[0])
                T = [j for j in P if any(k for k in [(s, i) for i in n]) in bytes(j) and j[0] != s]
                t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                if not t:
                    t = T
                while True:
                    if collections.Counter(M[mask])[c - 1] >= len(M[mask]) - n_words:
                        c += 1
                    if t:
                        p = list(random.sample(t, 1)[0])
                        m = np.array(transitional_p_matrix(p))
                        m[s][p[0]] = m[s][p[0]] + 1
                        if c not in M + m:
                            break
                        else:
                            t.remove(tuple(p))
                    else:
                        c += 1
                        T = [i for i in P if any(j for j in [(s, k) for k in n]) in bytes(i) and i[0] != s]
                        t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                        if not t:
                            t = T
            M += m
            v += p
            r = P.index(tuple(p))
            if tuple(p) in S[r]:
                S[r].remove(tuple(p))
            else:
                break
        if all(np.diagonal(M) == np.zeros(n_words)) and len(v) == n_words * n_repetitions:
            if M[mask].min() == math.floor(n_repetitions * n_words / len(M[mask])) - 1:
                if M[mask].max() == math.ceil(n_repetitions * n_words / len(M[mask])):
                    v.append(v[0])
                    M = np.array(transitional_p_matrix(v))
                    v.pop()
                    break
    return v, M


def compute_rhythmicity_index_sylls_stream(stream, patterns):
    count_patterns = []
    patterns = [tuple(pat) for pat in patterns]
    for feature_stream in list(zip(*[syllable.binary_features for syllable in stream])):
        c = 0
        for iSyll in range(len(feature_stream) - max(len(i) for i in patterns)):
            if any(i == feature_stream[iSyll: iSyll + len(i)] for i in patterns):
                c += 1
        count_patterns.append(c / (len(feature_stream) - max(len(i) for i in patterns)))

    return count_patterns


def get_oscillation_patterns(lag):
    kernel = [1] + [0] * (lag - 1) + [1] + [0] * (lag - 1)
    return [list(np.roll(kernel, i)) for i in range(lag)]


def overlap_matrix(words: List[Word]):
    n_words = len(words)
    n_sylls_per_word = len(words[0].syllables)

    oscillation_patterns = get_oscillation_patterns(lag=n_sylls_per_word)

    overlap = np.zeros([n_words, n_words])
    for i1, i2 in tqdm(list(itertools.product(range(n_words), range(n_words)))):
        word_pair_features = [f1 + f2 for f1, f2 in zip(words[i1].binary_features, words[i2].binary_features)]

        matches = 0
        for word_pair_feature in word_pair_features:
            if word_pair_feature in oscillation_patterns:
                matches += 1

        overlap[i1, i2] = matches

    return overlap


def read_ipa_seg_order_of_phonemes(
        ipa_seg_path: str = IPA_SEG_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[Phoneme], Dict[str, Phoneme]]:
    """
    Read order of phonemes, i.e. phonemes from a corpus together with the positions at which they
    appear in a bag of words.

    :param return_as_dict:
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

    if return_as_dict:
        return phonemes

    return phonemes.values()


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


def read_feature_syllables(
    phoneme_pattern: Union[str, list] = "cV",
    return_as_dict: bool = False
) -> Union[List[Syllable], Dict[str, Syllable]]:
    """Generate syllables form feature-phonemes. Only keep syllables that follow the phoneme pattern"""
    phonemes = read_phoneme_features(return_as_dict=True)

    logging.info("SELECT SYLLABLES WITH GIVEN PHONEME-TYPE PATTERN AND WITH PHONEMES WE HAVE FEATURES FOR")
    valid_phoneme_types = ["c", "C", "v", "V"]

    phoneme_types_user = list(phoneme_pattern) if isinstance(phoneme_pattern, str) else phoneme_pattern

    phoneme_types = list(filter(lambda p: p in valid_phoneme_types, phoneme_types_user))

    if phoneme_types_user != phoneme_types:
        logging.warning(f"ignoring invalid phoneme types {phoneme_types_user} -> {phoneme_types}. "
              f"You can use the following phoneme types in your pattern: {valid_phoneme_types}")

    logging.info(f"Search for phoneme-pattern '{''.join(phoneme_types)}'")

    labels_mapping = {"c": LABELS_C, "C": LABELS_C, "v": LABELS_V, "V": LABELS_V}
    syll_feature_labels = list(map(lambda t: labels_mapping[t], phoneme_types))

    single_consonants, multi_consonants, short_vowels, long_vowels = [], [], [], []
    for phoneme in phonemes.values():
        if phoneme.features[PHONEME_FEATURE_LABELS.index('cons')] == "+":
            if len(phoneme.id) == 1:
                single_consonants.append(phoneme.id)
            else:
                multi_consonants.append(phoneme.id)
        else:
            if len(phoneme.id) == 2:
                if phoneme.features[PHONEME_FEATURE_LABELS.index('long')] == "+":
                    long_vowels.append(phoneme.id)
                else:
                    short_vowels.append(phoneme.id)

    phonemes_mapping = {"c": single_consonants, "C": multi_consonants, "v": short_vowels, "V": long_vowels}

    phonemes_factors = list(map(lambda phoneme_type: phonemes_mapping[phoneme_type], phoneme_types))
    total_combs = reduce(lambda a, b: a * b, [len(phonemes_factor) for phonemes_factor in phonemes_factors])
    if total_combs > 100_000_000:
        logging.warning(f"Combinatorial explosion with {total_combs} combinations for '{phoneme_types}'."
              "This can happen when you use 'C' in your pattern.")

    syllables_phoneme_comb = {}
    for phon_tuple in itertools.product(*phonemes_factors):
        syll_id = "".join(phon_tuple)
        syll_phons = []
        syll_features = []
        for p, phoneme_feature_labels in zip(phon_tuple, syll_feature_labels):
            phoneme = phonemes[p]
            syll_phons.append(phoneme)
            for label in phoneme_feature_labels:
                if phoneme.features[PHONEME_FEATURE_LABELS.index(label)] == "+":
                    syll_features.append(1)
                else:
                    syll_features.append(0)
        syllables_phoneme_comb[syll_id] = Syllable(
            id=syll_id, info={}, phonemes=syll_phons, features=syll_features, custom_features=[]
        )

    if return_as_dict:
        return syllables_phoneme_comb

    return list(syllables_phoneme_comb.values())


def read_syllables_corpus(
        syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
    logging.info("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")

    with open(syllables_corpus_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    syllables_dict: Dict[str, Syllable] = {}

    for syll_stats in fdata[1:]:
        syll_ipa = phonecodes.xsampa2ipa(syll_stats[1], 'deu')
        info = {"freq": int(syll_stats[2]), "prob": float(syll_stats[3])}
        if syll_ipa not in syllables_dict or syllables_dict[syll_ipa].info != info:
            syllables_dict[syll_ipa] = Syllable(id=syll_ipa, phonemes=[], info=info, features=[], custom_features=[])
        else:
            logging.warning(
                f"Dropping syllable '{syll_ipa}' with conflicting stats {info} != {syllables_dict[syll_ipa].info}."
            )
            del syllables_dict[syll_ipa]

    if return_as_dict:
        return syllables_dict

    return syllables_dict.values()


def read_bigrams(
    ipa_bigrams_path: str = IPA_BIGRAMS_DEFAULT_PATH,
    return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
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
            bigrams_dict[bigram] = Syllable(id=bigram, phonemes=[], info=info, features=[], custom_features=[])
        else:
            logging.warning(
                f"Dropping bigram '{bigram}' with conflicting stats {info} != {bigrams_dict[bigram].info}."
            )
            del bigrams_dict[bigram]

    if return_as_dict:
        return bigrams_dict

    return bigrams_dict.values()


def read_trigrams(
        ipa_trigrams_path: str = IPA_TRIGRAMS_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
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
            trigrams_dict[trigram] = Syllable(id=trigram, phonemes=[], info=info, features=[], custom_features=[])
        else:
            logging.warning(
                f"Dropping trigram '{trigram}' with conflicting stats {info} != {trigrams_dict[trigram].info}."
            )
            del trigrams_dict[trigram]

    if return_as_dict:
        return trigrams_dict

    return trigrams_dict.values()


def export_speech_synthesiser(syllables: Iterable[Syllable]):
    logging.info("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
    syllables_dir = os.path.join(RESULTS_DEFAULT_PATH, "syllables")
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


def check_syll_feature_overlap(syllables):
    all_feats = [feat for syll in syllables for phon_feats in syll.phonotactic_features for feat in phon_feats]
    return len(all_feats) == len(set(all_feats))


def generate_subset_syllables(syllables, lookback_syllables):
    subset = []
    for new_syll in syllables:
        syll_test_set = lookback_syllables + [new_syll]
        if check_syll_feature_overlap(syll_test_set):
            subset.append(new_syll)

    return subset


def generate_words(syllables, n_sylls=3, n_look_back=2, max_tries=10_000) -> List:
    words = {}
    for _ in tqdm(range(max_tries)):
        sylls = []
        for _ in range(n_sylls):
            sub = generate_subset_syllables(syllables, sylls[-n_look_back:])
            if sub:
                syll = random.sample(sub, 1)[0]
                sylls.append(syll)

        if len(sylls) == n_sylls:
            word_id = "".join(s.id for s in sylls)
            word_features = list(zip(*[s.binary_features for s in sylls]))

            if word_id not in words:
                words[word_id] = Word(id=word_id, info={}, syllables=sylls, features=word_features)

    return list(words.values())


def filter_rare_onset_phonemes(syllables: Iterable[Syllable], phonemes: Dict[str, Phoneme],
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


def filter_common_phoneme_syllables(syllables, native_phonemes):
    return filter(lambda syll: all([(phon.id in native_phonemes) for phon in syll.phonemes]), syllables)


def filter_uniform_syllables(sylls):
    freqs = [s.info["freq"] for s in sylls]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    sylls = [s[0] for s in filter(lambda s: s[1] > 0.05, zip(sylls, p_vals_uniform))]
    return sylls


def check_bigram_stats(word: Word, valid_bigrams: List[Syllable]):
    phonemes = [phon for syll in word.syllables for phon in syll.phonemes]

    for phon_1, phon_2 in zip(phonemes[:-1], phonemes[1:]):
        if "".join([phon_1.id, phon_2.id]) not in valid_bigrams:
            return False

    return True


def check_trigram_stats(word: Word, valid_trigrams: List[str]):
    phonemes = [phon for syll in word.syllables for phon in syll.phonemes]

    for phon_1, phon_2, phon_3 in zip(phonemes[:-2], phonemes[1:-1], phonemes[2:]):
        if "".join([phon_1.id, phon_2.id, phon_3.id]) not in valid_trigrams:
            return False

    return True


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


def sample_min_overlap_lexicon(words, overlap, n_words=6, max_overlap=1, max_yields=10):
    # TODO make pretty
    overlap = np.array(overlap)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    logging.info(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0

    for max_pair_overlap, max_overlap_with_n_words in itertools.product(range(max_overlap + 1), range(1, math.comb(n_words, 2))):

        max_cum_overlap = max_pair_overlap*max_overlap_with_n_words

        if max_pair_overlap != 0:
            logging.warning(f"Increasing allowed overlaps: MAX_PAIRWISE_OVERLAP={max_pair_overlap}, MAX_CUM_OVERLAP={max_cum_overlap}")

        # WORDSxWORDS boolean matrix indicating if the words can be paired together
        # e.g. valid_word_pairs_matrix[0, 0] = False, bc. no word is parable with itself
        valid_word_pairs_matrix = (overlap <= max_pair_overlap)

        # represent the matrix from above as a list of pairs of word indexes, e.g. [[0, 1], [0, 2], ...]
        valid_pairs = set(frozenset(i) for i in zip(*np.where(valid_word_pairs_matrix)))

        def check_syll_pair(p):
            li = [s.id for i in p for s in words[i].syllables]
            return len(set(li)) == len(li)

        valid_pairs = set(filter(check_syll_pair, valid_pairs))

        for start_pair in valid_pairs:
            # logging.info(f"max overlap: {max_overlap}; start with pair: {i}/{len(valid_pairs)}")
            lexicon_indexes = set(start_pair)
            sum_overlaps = 0

            for candidate_idx in range(len(overlap)):
                if candidate_idx not in lexicon_indexes:
                    has_min_overlap = [({known_idx, candidate_idx} in valid_pairs) for known_idx in lexicon_indexes]

                    if all(has_min_overlap):
                        overlaps = [overlap[known_idx, candidate_idx] for known_idx in lexicon_indexes]

                        if sum(overlaps) <= (max_cum_overlap-sum_overlaps):
                            lexicon_indexes.add(candidate_idx)
                            sum_overlaps += sum(overlaps)

                            if len(lexicon_indexes) == n_words:
                                valid_sub_matrix = valid_word_pairs_matrix[np.ix_(list(lexicon_indexes), list(lexicon_indexes))]
                                assert np.all(np.int32(valid_sub_matrix) == (1 - np.eye(n_words)))
                                word_ids = []
                                word_objects = []
                                for index in lexicon_indexes:
                                    word_ids.append(words[index].id)
                                    word_objects.append(words[index])
                                lexicon = Lexicon(id="".join(word_ids), words=word_objects, info={"cumulative_overlap": sum_overlaps})

                                yield lexicon

                                yields += 1

                                if yields == max_yields:
                                    return


def filter_common_onset_words(words, sylls, native_phonemes):
    logging.info("EXCLUDE WORDS WITH LOW ONSET SYLLABLE PROBABILITY")
    rare_phonemes = filter_rare_onset_phonemes(sylls, native_phonemes)
    logging.info("Rare onset phonemes:", [p.id for p in rare_phonemes])

    # word[0][0] = first phoneme in first syllable of the word
    return filter(lambda word: word[0][0] not in rare_phonemes, words)


def filter_gram_stats(words, uniform_bigrams=False, uniform_trigrams=False):
    logging.info("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    bigrams = read_bigrams()
    trigrams = read_trigrams()

    if uniform_bigrams:
        bigrams = list(filter(lambda g: g.info["p_unif"] > 0.05, bigrams))

    if uniform_trigrams:
        trigrams = list(filter(lambda g: g.info["p_unif"] > 0.05, trigrams))

    list_bigrams = [b.id for b in bigrams]
    list_trigrams = [b.id for b in trigrams]

    return filter(
        lambda w: check_bigram_stats(w, list_bigrams) and check_trigram_stats(w, list_trigrams),
        words
    )


def sample_word_randomization(lexicon: Lexicon, randomized_word_indexes):
    for iRand in range(N_RANDOMIZATIONS_PER_STREAM):
        yield [lexicon.words[i] for i in randomized_word_indexes[iRand]]


def sample_syllable_randomization(lexicon: Lexicon, randomized_syllable_indexes):
    lexicon_syllables = [s for word in lexicon.words for s in word.syllables]

    for iRand in range(N_RANDOMIZATIONS_PER_STREAM):
        stream_syllable_randomized = [lexicon_syllables[i] for i in randomized_syllable_indexes[iRand]]

        yield stream_syllable_randomized


def check_rhythmicity(stream: Union[List[Word], List[Syllable]], max_ri=0.1):
    patterns = get_oscillation_patterns(N_SYLLABLES_PER_WORD)
    if isinstance(stream[0], Word):
        sylls_stream = [syll for word in stream for syll in word.syllables]
        rhythmicity_index = compute_rhythmicity_index_sylls_stream(sylls_stream, patterns)
    else:
        rhythmicity_index = compute_rhythmicity_index_sylls_stream(stream, patterns)
    if max(rhythmicity_index) <= max_ri:
        return stream, {"rhythmicity_indexes": rhythmicity_index}

    return None


def merge_with_corpus(feature_syllables, syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH):
    """
    Select syllables from the given corpus and add the binary_features from the feature syllables
    :param feature_syllables:
    :param syllables_corpus_path:
    :return:
    """
    corpus_syllables = read_syllables_corpus(syllables_corpus_path)
    if not isinstance(feature_syllables, dict):
        raise TypeError(f"Please make sure you provide the feature-syllables as a dict "
                        f"(e.g. read_syllables_with_phoneme_features('cV', return_as_dict=True)). "
                        f"This will make the matching go much faster")

    merged_syllables = []
    for syllable in corpus_syllables:
        if syllable.id in feature_syllables:
            merged_syllables.append(Syllable(
                id=syllable.id,
                info=syllable.info,
                phonemes=feature_syllables[syllable.id].phonemes,
                binary_features=feature_syllables[syllable.id].binary_features,
                custom_features=add_custom_features(feature_syllables[syllable.id])
            ))

    return merged_syllables


def add_custom_features(syllable):
    syll_feats = [[] for _ in syllable.phonemes]
    for i, phon in enumerate(syllable.phonemes):
        phon_feats = phon.features
        is_consonant = phon_feats[PHONEME_FEATURE_LABELS.index("cons")] == "+"
        if is_consonant:
            if phon_feats[SON] == '+':
                syll_feats[i].append("son")
            if phon_feats[SON] != '+' and phon_feats[CONT] != '+':
                syll_feats[i].append("plo")
            if phon_feats[SON] != '+' and phon_feats[CONT] == '+':
                syll_feats[i].append("fri")
            if phon_feats[LAB] == '+':
                syll_feats[i].append("lab")
            if phon_feats[COR] == '+' and phon_feats[HI] != '+':
                syll_feats[i].append("den")
            if "lab" not in syll_feats[i] and "den" not in syll_feats[i]:
                syll_feats[i].append("oth")
        else:
            for vowel in ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ø', 'y']:
                if vowel in phon.id:
                    syll_feats[i].append(vowel)

    return syll_feats


@maybe_load_from_file(path=os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle'), force_redo=False)
def generate_stream_randomization():
    logging.info("GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs")
    TP_struct_V = []
    for _ in tqdm(range(N_RANDOMIZATIONS_PER_STREAM)):
        while True:
            v_struct, m_struct = pseudo_rand_tp_struct()
            if v_struct not in TP_struct_V:
                TP_struct_V.append(v_struct)
                break

    TP_random_V = []
    for _ in tqdm(range(N_RANDOMIZATIONS_PER_STREAM)):
        while True:
            v_random, m_struct = pseudo_rand_tp_random()
            if v_random not in TP_random_V:
                TP_random_V.append(v_random)
                break

    return TP_struct_V, TP_random_V
