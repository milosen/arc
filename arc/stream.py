import csv
import itertools
import pickle
import random
from functools import reduce
from math import comb
from typing import Union, Any, Generator, Dict, Iterable

import itertools
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Any, Generator, List, Tuple, Dict, Set

from pydantic import BaseModel, ValidationError, PositiveInt
from scipy import stats
from tqdm.rich import tqdm

from arc.definitions import *
from arc.functional import *
from arc.phonecodes import phonecodes
from arc.types import *


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
    print("READ ORDER OF PHONEMES IN WORDS")
    fdata = list(csv.reader(open(ipa_seg_path, "r"), delimiter='\t'))
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
    print("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")

    with open(binary_features_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file))

    phons = [row[0] for row in fdata[1:]]
    feats = [row[1:] for row in fdata[1:]]

    phonemes_dict = {}
    for phon, features in zip(phons, feats):
        if phon not in phonemes_dict or features == phonemes_dict[phon].features:
            phonemes_dict[phon] = Phoneme(id=phon, features=features, order=[], info={})
        else:
            print(f"Warning: Dropping phoneme '{phon}' with conflicting feature entries {features} != {phonemes_dict[phon].features}.")
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

    print("SELECT SYLLABLES WITH GIVEN PHONEME-TYPE PATTERN AND WITH PHONEMES WE HAVE FEATURES FOR")
    valid_phoneme_types = ["c", "C", "v", "V"]

    phoneme_types_user = list(phoneme_pattern) if isinstance(phoneme_pattern, str) else phoneme_pattern

    phoneme_types = list(filter(lambda p: p in valid_phoneme_types, phoneme_types_user))

    if phoneme_types_user != phoneme_types:
        print(f"Warning: ignoring invalid phoneme types {phoneme_types_user} -> {phoneme_types}. "
              f"You can use the following phoneme types in your pattern: {valid_phoneme_types}")

    print(f"Search for phoneme-pattern '{''.join(phoneme_types)}'")

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
        print(f"Warning: Combinatorial explosion with {total_combs} combinations for '{phoneme_types}'."
              "This can happen when you use 'C' in your pattern.")

    syllables_phoneme_comb = {}
    for phon_tuple in itertools.product(*phonemes_factors):
        syll_id = "".join(phon_tuple)
        syllables_phoneme_comb[syll_id] = Syllable(
            id=syll_id, info={}, phonemes=[phonemes[p] for p in phon_tuple]
        )

    if return_as_dict:
        return syllables_phoneme_comb

    return list(syllables_phoneme_comb.values())


def read_syllables_corpus(
        syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
    print("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")

    with open(syllables_corpus_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    syllables_dict: Dict[str, Syllable] = {}

    for syll_stats in fdata[1:]:
        syll_ipa = phonecodes.xsampa2ipa(syll_stats[1], 'deu')
        info = {"freq": int(syll_stats[2]), "prob": float(syll_stats[3])}
        if syll_ipa not in syllables_dict or syllables_dict[syll_ipa].info != info:
            syllables_dict[syll_ipa] = Syllable(id=syll_ipa, phonemes=[], info=info)
        else:
            print(
                f"Warning: Dropping syllable '{syll_ipa}' with conflicting stats {info} != {syllables_dict[syll_ipa].info}.")
            del syllables_dict[syll_ipa]

    if return_as_dict:
        return syllables_dict

    return syllables_dict.values()


def read_bigrams(
    ipa_bigrams_path: str = IPA_BIGRAMS_DEFAULT_PATH,
    return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
    print("READ BIGRAMS")

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
            bigrams_dict[bigram] = Syllable(id=bigram, phonemes=[], info=info)
        else:
            print(
                f"Warning: Dropping bigram '{bigram}' with conflicting stats {info} != {bigrams_dict[bigram].info}.")
            del bigrams_dict[bigram]

    if return_as_dict:
        return bigrams_dict

    return bigrams_dict.values()


def read_trigrams(
        ipa_trigrams_path: str = IPA_TRIGRAMS_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
    print("READ TRIGRAMS")
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
            print(
                f"Warning: Dropping trigram '{trigram}' with conflicting stats {info} != {trigrams_dict[trigram].info}.")
            del trigrams_dict[trigram]

    if return_as_dict:
        return trigrams_dict

    return trigrams_dict.values()


def from_syllables_corpus(
        syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH,
        phoneme_pattern: Union[str, list] = "cV"
) -> Generator[Syllable, None, None]:

    syllables = read_syllables_corpus(syllables_corpus_path)

    valid_syllables = read_feature_syllables(phoneme_pattern, return_as_dict=True)

    for syllable in syllables:
        if syllable.id in valid_syllables:
            yield Syllable(
                id=syllable.id,
                info=syllable.info,
                phonemes=valid_syllables[syllable.id].phonemes
            )


def export_speech_synthesiser(syllables: Iterable[Syllable]):
    print("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
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


def generate_subset_sylls(iSyll, allIdx, f_Cls):
    """GENERATE A SUBSET OF SYLLABLES WITH NO OVERLAP OF CLASSES OF FEATURES WITH PREVIOUS SYLLABLES"""
    if iSyll in f_Cls[0][0]:
        set_1 = allIdx - f_Cls[0][0]
    elif iSyll in f_Cls[0][1]:
        set_1 = allIdx - f_Cls[0][1]
    elif iSyll in f_Cls[0][2]:
        set_1 = allIdx - f_Cls[0][2]
    if iSyll in f_Cls[1][0]:
        set_2 = allIdx - f_Cls[1][0]
    elif iSyll in f_Cls[1][1]:
        set_2 = allIdx - f_Cls[1][1]
    elif iSyll in f_Cls[1][2]:
        set_2 = allIdx - f_Cls[1][2]
    if iSyll in f_Cls[2][0]:
        set_3 = allIdx - f_Cls[2][0]
    elif iSyll in f_Cls[2][1]:
        set_3 = allIdx - f_Cls[2][1]
    elif iSyll in f_Cls[2][2]:
        set_3 = allIdx - f_Cls[2][2]
    elif iSyll in f_Cls[2][3]:
        set_3 = allIdx - f_Cls[2][3]
    elif iSyll in f_Cls[2][4]:
        set_3 = allIdx - f_Cls[2][4]
    elif iSyll in f_Cls[2][5]:
        set_3 = allIdx - f_Cls[2][5]
    elif iSyll in f_Cls[2][6]:
        set_3 = allIdx - f_Cls[2][6]
    elif iSyll in f_Cls[2][7]:
        set_3 = allIdx - f_Cls[2][7]
    set_i = set_1.intersection(set_1, set_2, set_3)
    return set_i


def maybe_load_from_file(path, force_redo: bool = False):
    def _outer_wrapper(wrapped_function):
        def _wrapper(*args, **kwargs):
            if not os.path.exists(path) or force_redo:
                data = wrapped_function(*args, **kwargs)

                with open(path, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

            else:
                print("SKIPPING EXECUTION GENERATION. LOADING DATA FROM FILE")
                with open(os.path.join(RESULTS_DEFAULT_PATH, "words.pickle"), 'rb') as f:
                    data = pickle.load(f)

            return data
        return _wrapper
    return _outer_wrapper


@maybe_load_from_file(path=os.path.join(RESULTS_DEFAULT_PATH, "words.pickle"), force_redo=False)
def generate_nsyll_words(syllables, f_cls, n_sylls=3, n_look_back=2, n_tries=1_000_000) -> List:
    print("GENERATE LIST OF TRISYLLABIC WORDS WITH NO OVERLAP OF COMPLEX PHONETIC FEATURES ACROSS SYLLABLES")
    words = set()
    all_idx = set(range(len(syllables)))
    # idx_sets = deque([all_idx]*n_look_back)
    for _ in tqdm(range(n_tries)):
        while True:
            syl_1 = random.sample(all_idx, 1)[0]
            set_1 = generate_subset_sylls(syl_1, all_idx, f_cls)
            if set_1:
                syl_2 = random.sample(list(set_1), 1)[0]
                set_2 = generate_subset_sylls(syl_2, all_idx, f_cls)
                if set_2:
                    set_3 = set_1.intersection(set_1, set_2)
                    if set_3:
                        syl_3 = random.sample(list(set_3), 1)[0]
                        break
        word = tuple([syl_1, syl_2, syl_3])
        words.add(word)

    words_list = []
    for w in words:
        ss = [syllables[idx] for idx in w]
        words_list.append(Word(id="".join(s.id for s in ss), info={}, syllables=ss))

    return words_list


def filter_rare_onset_phonemes(syllables: Iterable[Syllable], phonemes: Dict[str, Phoneme],
                               p_threshold: float = 0.05):
    print("FIND SYLLABLES THAT ARE RARE AT THE ONSET OF A WORD")

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


def has_valid_gram_stats(word_obj, valid_bigrams, valid_trigrams):
    """TODO: Change implementation for other syllables-settings?"""
    """Assuming that the individual syllables in the word are valid, we still have to check if the 
    bigrams and trigrams that have formed at the syllable transitions are valid.

    Example:
    word = "heː|pɛː|naː"
    valid_sylls = ["heː", "pɛː", "naː"]
    So, in addition, we have to check that the bigrams "eːp" and "ɛː|n" at the transitions are contained 
    in the valid bigrams list
    """
    valid = True
    word = word_obj.id
    seg_1 = word[1:4]
    seg_2 = word[4:7]
    if seg_1 not in valid_bigrams or seg_2 not in valid_bigrams:
        valid = False
    seg_1 = word[0:4]
    seg_2 = word[1:6]
    seg_3 = word[3:7]
    seg_4 = word[4:9]
    if any(i not in valid_trigrams for i in [seg_1, seg_2, seg_3, seg_4]):
        valid = False
    return valid


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

    print("LOAD WORDS FROM CSV FILE AND SELECT THOSE THAT CANNOT BE MISTAKEN FOR GERMAN WORDS")
    with open(os.path.join(RESULTS_DEFAULT_PATH, "words.csv"), 'r') as f:
        fdata = list(csv.reader(f, delimiter='\t'))
    rows = [row[0].split(",") for row in fdata]
    words = [row[0] for row in rows if row[1] == "0"]

    return words


def sample_min_overlap_lexicon(words, overlap, n_words=6, max_overlap=1, max_yields=10):
    overlap = np.array(overlap)
    options = dict((k, v) for k, v in locals().items() if not k == 'words' and not k == 'overlap')
    print(f"GENERATE MIN OVERLAP LEXICONS WITH OPTIONS {options}")
    yields = 0

    for max_pair_overlap, max_overlap_with_n_words in itertools.product(range(max_overlap + 1), range(1, comb(n_words, 2))):

        max_cum_overlap = max_pair_overlap*max_overlap_with_n_words

        if max_pair_overlap == 0:
            print(f"Trying zero overlap")
        else:
            print(f"Warning: Increasing allowed overlaps: MAX_PAIRWISE_OVERLAP={max_pair_overlap}, MAX_CUM_OVERLAP={max_cum_overlap}")

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
            # print(f"max overlap: {max_overlap}; start with pair: {i}/{len(valid_pairs)}")
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


@maybe_load_from_file(path=os.path.join(RESULTS_DEFAULT_PATH, "words_filtered.pickle"), force_redo=True)
def filter_words(words, sylls, native_phonemes):
    print("EXCLUDE WORDS WITH LOW ONSET SYLLABLE PROBABILITY")
    rare_phonemes = filter_rare_onset_phonemes(sylls, native_phonemes)
    print("Rare onset phonemes:", [p.id for p in rare_phonemes])

    # w[0][0] = first phoneme in first syllable if word w
    words = list(filter(lambda w: w[0][0] not in rare_phonemes, tqdm(words)))

    list_bigrams = [b.id for b in bigrams]
    list_trigrams = [b.id for b in trigrams]

    print("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    words = list(filter(lambda w: has_valid_gram_stats(w, list_bigrams, list_trigrams), tqdm(words)))

    return words


def sample_word_randomization(lexicon: Lexicon, randomized_word_indexes):
    for iRand in range(N_RANDOMIZATIONS_PER_STREAM):

        rand_words_lexicon = [lexicon.words[i] for i in randomized_word_indexes[iRand]]
        stream_word_randomized = [s.id for word in rand_words_lexicon for s in word.syllables]

        yield stream_word_randomized


def sample_syllable_randomization(lexicon: Lexicon, randomized_syllable_indexes):
    lexicon_syllables = [s.id for word in lexicon.words for s in word.syllables]

    for iRand in range(N_RANDOMIZATIONS_PER_STREAM):
        stream_syllable_randomized = [lexicon_syllables[i] for i in randomized_syllable_indexes[iRand]]

        yield stream_syllable_randomized


def check_rhythmicity(stream, patterns, feats, max_ri=0.1):
    rhythmicity_index = compute_rhythmicity_index(stream, patterns, feats)
    if max(rhythmicity_index) <= max_ri:
        return stream, {"rhythmicity_indexes": rhythmicity_index}

    return None


def merge_with_corpus(feature_syllables, syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH):
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
                phonemes=feature_syllables[syllable.id].phonemes
            ))

    return merged_syllables


def syllabic_features_from_list(syllables_list: List[Syllable]) -> List[Set[PositiveInt]]:
    i_son, i_plo, i_fri, i_lab, i_den, i_oth, idx_a, idx_e, idx_i, idx_o, idx_u, idx_ae, idx_oe, idx_ue \
        = tuple([] for _ in range(14))

    for i, syll in enumerate(syllables_list):
        onset_phoneme_features = syll[0].features

        if onset_phoneme_features[SON] == '+':
            i_son.append(i)
        if onset_phoneme_features[SON] != '+' and onset_phoneme_features[CONT] != '+':
            i_plo.append(i)
        if onset_phoneme_features[SON] != '+' and onset_phoneme_features[CONT] == '+':
            i_fri.append(i)
        if onset_phoneme_features[LAB] == '+':
            i_lab.append(i)
        if onset_phoneme_features[COR] == '+' and onset_phoneme_features[HI] != '+':
            i_den.append(i)
        if i not in i_lab and i not in i_den:
            i_oth.append(i)
        if 'a' in syll.id:
            idx_a.append(i)
        if 'e' in syll.id:
            idx_e.append(i)
        if 'i' in syll.id:
            idx_i.append(i)
        if 'o' in syll.id:
            idx_o.append(i)
        if 'u' in syll.id:
            idx_u.append(i)
        if 'ɛ' in syll.id:
            idx_ae.append(i)
        if 'ø' in syll.id:
            idx_oe.append(i)
        if 'y' in syll.id:
            idx_ue.append(i)

    f_manner = [set(i_son), set(i_plo), set(i_fri)]
    f_place = [set(i_oth), set(i_lab), set(i_den)]
    f_vowel = [set(idx_a), set(idx_e), set(idx_i), set(idx_o), set(idx_u), set(idx_ae), set(idx_oe), set(idx_ue)]
    syllabic_features = [f_manner, f_place, f_vowel]

    return syllabic_features


def read_binary_features(binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH) -> BinaryFeatures:
    print("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")
    fdata = list(csv.reader(open(binary_features_path, "r")))
    labels = fdata[0][1:]
    phons = [i[0] for i in fdata[1:]]
    numbs = [i[1:] for i in fdata[1:]]

    consonants = []
    for phon, numb in zip(phons, numbs):
        if numb[labels.index('cons')] == '+':
            consonants.append(phon)

    long_vowels = []
    for phon, numb in zip(phons, numbs):
        if numb[labels.index('long')] == '+' and phon not in consonants:
            long_vowels.append(phon)

    bin_feats = BinaryFeatures(
        labels=labels,
        labels_c=LABELS_C,
        labels_v=LABELS_V,
        phons=phons,
        numbs=numbs,
        consonants=consonants,
        long_vowels=long_vowels,
        n_features=(len(LABELS_C) + len(LABELS_V))
    )

    return bin_feats


if __name__ == '__main__':
    feature_syllables = read_feature_syllables("cV", return_as_dict=True)

    sylls = merge_with_corpus(feature_syllables)

    freqs = [s.info["freq"] for s in sylls]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))
    sylls = [s[0] for s in filter(lambda s: s[1] > 0.05, zip(sylls, p_vals_uniform))]

    native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)
    sylls = list(filter(lambda syll: all([(phon.id in native_phonemes) for phon in syll.phonemes]), sylls))

    export_speech_synthesiser(sylls)

    bigrams = read_bigrams()
    trigrams = read_trigrams()

    bigrams_uniform = filter(lambda g: g.info["p_unif"] > 0.05, bigrams)
    trigrams_uniform = filter(lambda g: g.info["p_unif"] > 0.05, trigrams)

    syllabic_features = syllabic_features_from_list(sylls)

    REGENERATE_WORDS = True
    CHECK_VALID_GERMAN = False
    REFILTER_WORDS = True
    REGENERATE_STREAM_RANDOMIZATION = False

    words: List[Word] = generate_nsyll_words(sylls, syllabic_features)

    words: List[Word] = filter_words(words, sylls, native_phonemes)

    print(len(words))

    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)

    # KERNELS SIMULATING A STATIONARY OSCILLATORY SIGNAL AT THE FREQUENCY OF INTEREST (LAG = 3)
    oscillation_patterns = [tuple([i for i in np.roll((1, 0, 0, 1, 0, 0), j)]) for j in range(N_SYLLABLES_PER_WORD)]

    print("EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR")
    features = list(map(lambda w: binary_feature_matrix(w, bin_feat), tqdm(words)))

    print("compute word overlap matrix")
    overlap = compute_word_overlap_matrix(words=words, features=features, oscillation_patterns=oscillation_patterns)

    if not os.path.exists(
            os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle')) or REGENERATE_STREAM_RANDOMIZATION:
        # GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs
        TP_struct_V = [];
        TP_random_V = []
        while len(TP_struct_V) < N_RANDOMIZATIONS_PER_STREAM:
            v_struct, M_struct = pseudo_rand_TP_struct()
            v_random, M_struct = pseudo_rand_TP_random()
            if v_struct not in TP_struct_V and v_random not in TP_random_V:
                TP_struct_V.append(v_struct)
                TP_random_V.append(v_random)

        fdata = [TP_struct_V, TP_random_V]

        with open(os.path.join(RESULTS_DEFAULT_PATH, 'random_streams_indexes.pickle'), 'wb') as f:
            pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("SKIPPING RANDOM STREAM GENERATION. LOADING STREAMS FROM FILE")
        with open(os.path.join(RESULTS_DEFAULT_PATH, "random_streams_indexes.pickle"), 'rb') as f:
            fdata = pickle.load(f)
            randomized_word_indexes = fdata[0]
            randomized_syllable_indexes = fdata[1]

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    s1w = None
    for lexicon_1 in lexicon_generator_1:
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat, max_ri=0.09)

            if s1w:
                break

        if not s1w:
            continue

        break

    if s1w:
        stream, info = s1w
        print("Stream 1: ", "".join(syllable for syllable in stream))
        print("Rhythmicity Indexes 1: ", info["rhythmicity_indexes"])
        print("Found lexicon: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"])
    else:
        print("Nothing found")

    exit()

    s1w, s2w = None, None
    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    # print pairwise lexicon generation
    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.id for lexicon in [lexicon_1, lexicon_2] for word in lexicon.words for s in word.syllables]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"], lexicon_2.id,
              lexicon_2.info["cumulative_overlap"])

        s1w = None
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat)

            if s1w:
                break

        if not s1w:
            continue

        s2w = None
        for stream_2_word_randomized in sample_word_randomization(lexicon_2, randomized_word_indexes):
            s2w = check_rhythmicity(stream_2_word_randomized, oscillation_patterns, bin_feat)

            if s2w:
                break

        if not s2w:
            continue

        break

    if s1w and s2w:
        stream1, stream_info1 = s1w
        stream2, stream_info2 = s2w
        print("Stream 1: ", "".join(syllable for syllable in stream1))
        print("Stream 2: ", "".join(syllable for syllable in stream2))
        print("Rhythmicity Indexes 1: ", stream_info1["rhythmicity_indexes"])
        print("Rhythmicity Indexes 2: ", stream_info2["rhythmicity_indexes"])
        print("Found compatible lexicons: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"], lexicon_2.id,
              lexicon_2.info["cumulative_overlap"])
    else:
        print("Nothing found")

    exit()

    print("SELECT TP STRUCT STREAMS AND TP RANDOM STREAMS WITH MINIMUM RHYTHMICITY INDEX")

    s1w = None
    s1s = None
    s2w = None
    s2s = None

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.id for lexicon in [lexicon_1, lexicon_2] for word in lexicon.words for s in word.syllables]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", lexicon_1.id, lexicon_2.id)

        # generate good-RI streams
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat)
            if s1w:
                break

        if not s1w:
            for stream in [s1w, s1s, s2w, s2s]:
                print(stream)
            continue

        for stream_2_word_randomized in sample_word_randomization(lexicon_2, randomized_word_indexes):
            s2w = check_rhythmicity(stream_2_word_randomized, oscillation_patterns, bin_feat)
            if s2w:
                break

        if not s2w:
            for stream in [s1w, s1s, s2w, s2s]:
                print(stream)
            continue

        for stream_1_syll_randomized in sample_syllable_randomization(lexicon_1, randomized_syllable_indexes):
            s1s = check_rhythmicity(stream_1_syll_randomized, oscillation_patterns, bin_feat)
            if s1s:
                break

        if not s1s:
            for stream in [s1w, s1s, s2w, s2s]:
                print(stream)
            continue

        for stream_2_syll_randomized in sample_syllable_randomization(lexicon_2, randomized_syllable_indexes):
            s2s = check_rhythmicity(stream_2_syll_randomized, oscillation_patterns, bin_feat)
            if s2s:
                break

        if all([s1w, s1s, s2w, s2s]):
            break

    for stream in [s1w, s1s, s2w, s2s]:
        print(stream)

    # SAVE LEXICON TRIALS
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'streams.pickle')
    fdata = [s1w, s1s, s2w, s2s]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)
