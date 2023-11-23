import itertools
import pickle
from functools import reduce
from typing import Union, Any, Generator, Dict

from arc.io import *
# from arc.syllables import select_consonant_vowel, select_occurrence_probability, export_speech_synthesiser

import test_pydantic as tp


example_struct = {
    "ka:fu:ry": {
        "ka:": {
            "k": {"order": None, "features": None},
            "a:": {"order": None, "features": None},
            "_info": {"freq": int, "prob": float, "p_unif": float}
        },
        "fu:": {
            "f": {"order": None, "features": None},
            "u:": {"order": None, "features": None},
            "_info": {"freq": int, "prob": float, "p_unif": float}
        },
        "ry:": {
            "r": {"order": None, "features": None},
            "y:": {"order": None, "features": None},
            "_info": {"freq": int, "prob": float, "p_unif": float}
        },
        "_info": {}
    },
    "...": {},
    "_info": {},
}


class DictionaryARC(dict):
    """Like a python dictionary, just with some modified extra methods"""
    def __init__(self, seq=None, **kwargs):
        super().__init__({} if seq is None else seq, **kwargs)
        if "_info" not in self:
            self["_info"] = {}

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __str__(self):
        keys_set = self.get_collection()
        return keys_set.__str__()

    def get_collection(self):
        return set(key for key in self.keys() if key != "_info")

    def get_info(self) -> dict:
        return self["_info"]

    def get_vals(self, key: str) -> Generator:
        for entry, data in self.items():
            yield entry, data["_info"][key]


def read_ipa_seg_order_of_phonemes(
        ipa_seg_path: str = IPA_SEG_DEFAULT_PATH,
        return_as_dict: bool = False
) -> Union[Iterable[tp.Phoneme], Dict[str, tp.Phoneme]]:
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
                phonemes[phon] = tp.Phoneme(id=phon, order=[position_in_word], features=[], info={})

    if return_as_dict:
        return phonemes

    return phonemes.values()


def read_phoneme_features(
        binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH,
        return_as_dict: bool = False,

) -> Union[Iterable[tp.Phoneme], Dict[str, tp.Phoneme]]:
    print("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")

    with open(binary_features_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file))

    phons = [i[0] for i in fdata[1:]]
    feats = [i[1:] for i in fdata[1:]]

    phonemes_dict = {}
    for phon, features in zip(phons, feats):
        if phon not in phonemes_dict or features == phonemes_dict[phon].features:
            phonemes_dict[phon] = tp.Phoneme(id=phon, features=features, order=[], info={})
        else:
            print(f"Warning: Dropping phoneme '{phon}' with conflicting feature entries {features} != {phonemes_dict[phon].features}.")
            del phonemes_dict[phon]

    if return_as_dict:
        return phonemes_dict

    return phonemes_dict.values()


def read_syllables_with_phoneme_features(
    phoneme_pattern: Union[str, list] = "cV",
    return_as_dict: bool = False
):
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
        syllables_phoneme_comb[syll_id] = tp.Syllable(
            id=syll_id, info={}, phonemes=[phonemes[p] for p in phon_tuple]
        )

    if return_as_dict:
        return syllables_phoneme_comb

    return syllables_phoneme_comb.values()


def read_syllables_corpus(syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH, return_as_dict: bool = False):
    print("READ SYLLABLES, FREQUENCIES AND PROBABILITIES FROM CORPUS AND CONVERT SYLLABLES TO IPA")

    with open(syllables_corpus_path, "r") as csv_file:
        fdata = list(csv.reader(csv_file, delimiter='\t'))

    syllables_dict: Dict[str, tp.Syllable] = {}

    freqs = [int(syll_stats[2]) for syll_stats in fdata[1:]]
    p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

    for syll_stats, p_val_uniform in zip(fdata[1:], p_vals_uniform):
        syll_ipa = phonecodes.xsampa2ipa(syll_stats[1], 'deu')
        info = {"freq": int(syll_stats[2]), "prob": float(syll_stats[3]), "p_unif": p_val_uniform}
        if syll_ipa not in syllables_dict:
            syllables_dict[syll_ipa] = tp.Syllable(id=syll_ipa, phonemes=[], info=info)
        else:
            print(
                f"Warning: Dropping syllable '{syll_ipa}' with conflicting stats {info} != {syllables_dict[syll_ipa]['_info']}.")
            del syllables_dict[syll_ipa]

    if return_as_dict:
        return syllables_dict

    return syllables_dict.values()


def from_syllables_corpus(
        syllables_corpus_path: str = SYLLABLES_DEFAULT_PATH,
        phoneme_pattern: Union[str, list] = "cV"
) -> Generator[tp.Syllable, None, None]:

    syllables = read_syllables_corpus(syllables_corpus_path)

    valid_syllables = read_syllables_with_phoneme_features(phoneme_pattern, return_as_dict=True)

    for syllable in syllables:
        if syllable.id in valid_syllables:
            yield tp.Syllable(
                id=syllable.id,
                info=syllable.info,
                phonemes=valid_syllables[syllable.id].phonemes
            )


def export_speech_synthesiser(syllables: Iterable[tp.Syllable]):
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


if __name__ == '__main__':

    sylls = from_syllables_corpus()

    native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)

    native_sylls = filter(lambda s: all([(phon.id in native_phonemes) for phon in syll.phonemes]), sylls)

    for syll in sylls:
        print(f"{syll} has native phonemes: ", all([(phon.id in native_phonemes) for phon in syll.phonemes]))
        # print(f"{syll} has native phonemes: ", syll in list(native_sylls))

    # export_speech_synthesiser(sylls)
