import itertools
import pickle
import random
from functools import reduce
from typing import Union, Any, Generator, Dict

from arc.io import *

import itertools
import pickle
from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Any, Generator, List, Tuple, Dict, Set

from pydantic import BaseModel, ValidationError, PositiveInt

from arc.functional import binary_feature_matrix, compute_word_overlap_matrix, map_iterable, pseudo_rand_TP_struct, \
    pseudo_rand_TP_random, compute_rhythmicity_index


class BaseDictARC(ABC):
    def full_str(self):
        """recursive string representation"""
        full_str = "["
        for entry in self.decompose():
            full_str += entry.full_str()
            full_str += ", "
        full_str = full_str[:-2]
        full_str += "]"
        return f"{self.__str__()} -> {full_str}"

    def as_dict(self):
        """recursive dict representation"""
        full_dict = {}
        full_list = []
        for entry in self.decompose():
            next_level = entry.as_dict()
            if isinstance(next_level, str):
                full_list.append(next_level)
            else:
                full_dict.update(**next_level)

        return {self.__str__(): full_dict or full_list}

    def get_vals(self, key: str) -> List:
        return [entry.info[key] for entry in self.decompose()]

    def __getitem__(self, item):
        return self.decompose()[item]

    @abstractmethod
    def decompose(self):
        pass


class Phoneme(BaseModel, BaseDictARC):
    id: str
    info: Dict[str, Any]
    order: List[PositiveInt]
    features: List[str]

    def __str__(self):
        return self.id

    def full_str(self):
        return self.__str__()

    def decompose(self):
        return []

    def as_dict(self):
        return self.id


class Syllable(BaseModel, BaseDictARC):
    id: str
    phonemes: List[Phoneme]
    info: Dict[str, Any]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.phonemes


class Word(BaseModel, BaseDictARC):
    id: str
    syllables: List[Syllable]
    info: Dict[str, Any]

    def __str__(self):
        return self.id

    def decompose(self):
        return self.syllables


class Lexicon(BaseModel, BaseDictARC):
    id: str
    words: List[Word]
    info: Dict[str, Any]

    def __str__(self):
        return [w.id for w in self.words].__str__()

    def decompose(self):
        return self.words


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

    phons = [i[0] for i in fdata[1:]]
    feats = [i[1:] for i in fdata[1:]]

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


def read_syllables_with_phoneme_features(
    phoneme_pattern: Union[str, list] = "cV",
    return_as_dict: bool = False
)-> Union[Iterable[Syllable], Dict[str, Syllable]]:
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

    return syllables_phoneme_comb.values()


def read_syllables_corpus(
        syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[Syllable], Dict[str, Syllable]]:
    print("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")

    with open(syllables_corpus_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    syllables_dict: Dict[str, Syllable] = {}

    freqs = [int(syll_stats[2]) for syll_stats in fdata[1:]]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    for syll_stats, p_val_uniform in zip(fdata[1:], p_vals_uniform):
        syll_ipa = phonecodes.xsampa2ipa(syll_stats[1], 'deu')
        info = {"freq": int(syll_stats[2]), "prob": float(syll_stats[3]), "p_unif": p_val_uniform}
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

    valid_syllables = read_syllables_with_phoneme_features(phoneme_pattern, return_as_dict=True)

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
        synth_string = '<phoneme alphabet="ipa" ph=' + '"' + syllable.syll + '"' + '>' + text + '</phoneme>'
        with open(os.path.join(syllables_dir, f'{str(syllable.syll[0:2])}.txt'), 'w') as f:
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


def generate_nsyll_words(syllables, f_cls, n_sylls=3, n_look_back=2, n_tries=1_000_000) -> List:
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
            li = [s.syll for i in p for s in words[i]]
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
                                lexicon = set(map(lambda index: words[index], lexicon_indexes))

                                yield lexicon, {"cumulative_overlap": sum_overlaps}

                                yields += 1

                                if yields == max_yields:
                                    return


if __name__ == '__main__':

    sylls = list(from_syllables_corpus())

    native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)

    native_sylls = filter(lambda s: all([(phon.id in native_phonemes) for phon in syll.phonemes]), sylls)

    for syll in sylls:
        print(f"{syll} has native phonemes: ", all([(phon.id in native_phonemes) for phon in syll.phonemes]))
        # print(f"{syll} has native phonemes: ", syll in list(native_sylls))

    # export_speech_synthesiser(sylls)

    bigrams = read_bigrams()

    trigrams = read_trigrams()

    print("SELECT BIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    bigrams_uniform = filter(lambda g: g.info["p_unif"] > 0.05, bigrams)

    print("SELECT TRIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    trigrams_uniform = filter(lambda g: g.info["p_unif"] > 0.05, trigrams)

    labls = PHONEME_FEATURE_LABELS

    i_Son, i_Plo, i_Fri, i_Lab, i_Den, i_Oth, idx_A, idx_E, idx_I, idx_O, idx_U, idx_AE, idx_OE, idx_UE \
        = tuple([] for _ in range(14))

    for i, syll in enumerate(sylls):
        onset_phoneme_features = syll[0].features

        if onset_phoneme_features[labls.index('son')] == '+':
            i_Son.append(i)
        if onset_phoneme_features[labls.index('son')] != '+' and onset_phoneme_features[labls.index('cont')] != '+':
            i_Plo.append(i)
        if onset_phoneme_features[labls.index('son')] != '+' and onset_phoneme_features[labls.index('cont')] == '+':
            i_Fri.append(i)
        if onset_phoneme_features[labls.index('lab')] == '+':
            i_Lab.append(i)
        if onset_phoneme_features[labls.index('cor')] == '+' and onset_phoneme_features[labls.index('hi')] != '+':
            i_Den.append(i)
        if i not in i_Lab and i not in i_Den:
            i_Oth.append(i)
        if 'a' in syll.id:
            idx_A.append(i)
        if 'e' in syll.id:
            idx_E.append(i)
        if 'i' in syll.id:
            idx_I.append(i)
        if 'o' in syll.id:
            idx_O.append(i)
        if 'u' in syll.id:
            idx_U.append(i)
        if 'ɛ' in syll.id:
            idx_AE.append(i)
        if 'ø' in syll.id:
            idx_OE.append(i)
        if 'y' in syll.id:
            idx_UE.append(i)

    f_manner = [set(i_Son), set(i_Plo), set(i_Fri)]
    f_place = [set(i_Oth), set(i_Lab), set(i_Den)]
    f_vowel_identity = [set(idx_A), set(idx_E), set(idx_I), set(idx_O), set(idx_U),
                        set(idx_AE), set(idx_OE), set(idx_UE)]
    syllabic_features = [f_manner, f_place, f_vowel_identity]

    syllables_list = [s.id for s in sylls]

    REGENERATE_WORDS = False
    CHECK_VALID_GERMAN = False

    if not os.path.exists(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv')) or REGENERATE_WORDS:
        print("GENERATE LIST OF TRISYLLABIC WORDS WITH NO OVERLAP OF COMPLEX PHONETIC FEATURES ACROSS SYLLABLES")
        # words = list(set([generate_trisyll_word(CVsyl, f_Cls) for _ in tqdm(range(1_000_000))]))
        words: List[Word] = generate_nsyll_words(sylls, syllabic_features)
        print("Words generated: ", len(words))

        # SAVE WORDS IN ONE CSV FILE
        with open(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv'), 'w') as f:
            writer = csv.writer(f)
            for word in words:
                writer.writerows([[word.id, 0]])

        with open(os.path.join(RESULTS_DEFAULT_PATH, 'words.pickle'), 'wb') as f:
            pickle.dump(words, f, pickle.HIGHEST_PROTOCOL)

    else:
        print("SKIPPING WORD GENERATION. LOADING WORDS FROM FILE")
        with open(os.path.join(RESULTS_DEFAULT_PATH, "words.pickle"), 'rb') as f:
            words: WordsList = pickle.load(f)

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

    # print("LOAD WORDS FROM CSV FILE AND SELECT THOSE THAT CANNOT BE MISTAKEN FOR GERMAN WORDS")
    # with open(os.path.join(RESULTS_DEFAULT_PATH, "words.csv"), 'r') as f:
    #     fdata = list(csv.reader(f, delimiter='\t'))
    # rows = [row[0].split(",") for row in fdata]
    # words = [row[0] for row in rows if row[1] == "0"]

    REFILTER_WORDS = False

    if not os.path.exists(os.path.join(RESULTS_DEFAULT_PATH, 'words_filtered.csv')) or REFILTER_WORDS:
        print("EXCLUDE WORDS WITH LOW ONSET SYLLABLE PROBABILITY")
        rare_phonemes = filter_rare_onset_phonemes(sylls, native_phonemes)
        print("Rare onset phonemes:", [p.id for p in rare_phonemes])

        # w[0][0] = first phoneme in first syllable if word w
        words = list(filter(lambda w: w[0][0] not in rare_phonemes, tqdm(words)))

        list_bigrams = [b.id for b in bigrams]
        list_trigrams = [b.id for b in trigrams]

        print("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
        words = list(filter(lambda w: has_valid_gram_stats(w, list_bigrams, list_trigrams), tqdm(words)))

        # SAVE WORDS IN ONE CSV FILE
        with open(os.path.join(RESULTS_DEFAULT_PATH, 'words_filtered.csv'), 'w') as f:
            writer = csv.writer(f)
            for word in words:
                writer.writerows([[word.id, 0]])

        with open(os.path.join(RESULTS_DEFAULT_PATH, 'words_filtered.pickle'), 'wb') as f:
            pickle.dump(words, f, pickle.HIGHEST_PROTOCOL)

    else:
        print("SKIPPING WORD GENERATION. LOADING WORDS FROM FILE")
        with open(os.path.join(RESULTS_DEFAULT_PATH, "words_filtered.pickle"), 'rb') as f:
            words: List[Word] = pickle.load(f)

    print(len(words))

    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)

    # KERNELS SIMULATING A STATIONARY OSCILLATORY SIGNAL AT THE FREQUENCY OF INTEREST (LAG = 3)
    oscillation_patterns = [tuple([i for i in np.roll((1, 0, 0, 1, 0, 0), j)]) for j in range(N_SYLLABLES_PER_WORD)]

    print("EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR")
    features = list(map(lambda w: binary_feature_matrix(w, bin_feat), tqdm(words)))

    print("compute_word_overlap_matrix")
    overlap = compute_word_overlap_matrix(words=words, features=features, oscillation_patterns=oscillation_patterns)

    REGENERATE_STREAM_RANDOMIZATION = True
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

    # SELECT TP STRUCT STREAMS AND TP RANDOM STREAMS WITH MINIMUM RHYTHMICITY INDEX

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    # print pairwise lexicon generation
    for (lexicon_1, info_1), (lexicon_2, info_2) in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.syll for lexicon in [lexicon_1, lexicon_2] for word in lexicon for s in word]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", "".join(s.id for word in lexicon_1 for s in word), "".join(s.id for word in lexicon_2 for s in word))

    exit()

    # print lexicon and stream for test
    for lexicon_1, info_1 in lexicon_generator_1:

        print(extract_lexicon_string(lexicon_1), "Cumulative overlap:", info_1["cumulative_overlap"])

        s1w = None
        for stream_1_word_randomized in sample_word_randomization(lexicon_1, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat)

            if s1w:
                break

        if s1w:
            print("Lexicon:", extract_lexicon_string(lexicon_1))
            print("Cumulative overlap of words in lexicon:", info_1["cumulative_overlap"])

            stream, stream_info = s1w
            print("Stream: ", "".join(syllable for syllable in stream))
            print("Rhythmicity Indexes: ", stream_info["rhythmicity_indexes"])
            break

    exit()

    s1w = None
    s1s = None
    s2w = None
    s2s = None

    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.syll for lexicon in [lexicon_1, lexicon_2] for word in lexicon for s in word]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", extract_lexicon_string(lexicon_1), extract_lexicon_string(lexicon_2))

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
