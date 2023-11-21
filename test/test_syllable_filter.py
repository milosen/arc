from arc.io import *
from arc.syllables import select_consonant_vowel


def old_read_phonemes(ipa_seg_path: str = IPA_SEG_DEFAULT_PATH) -> PhonemesList:
    print("READ ORDER OF PHONEMES IN WORDS")
    fdata = list(csv.reader(open(ipa_seg_path, "r"), delimiter='\t'))
    phonemes = []
    for phon_data in tqdm(fdata[1:]):
        phon_data_split = phon_data[0].split(",")
        if len(phon_data_split) == 3:
            phon = phon_data_split[1].replace('"', '').replace("g", "ɡ")
            order = int(phon_data_split[2])
            phonemes.append(Phoneme(phon=phon, order=order))
    return phonemes


def new_read_phonemes(binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH) -> BinaryFeatures:
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


def new_select_consonant_vowel(
        syllables: SyllablesList,
        phoneme_pattern: str = "cVb",  # e.g. cV, Cv etc.  (c/C=single-character-/multi-character-consonant, v/V=long/short vowel
) -> SyllablesList:
    """Only select those syllables that follow the phoneme pattern"""
    print("SELECT CONSONANT-VOWEL SYLLABLES WITH LONG VOWEL LENGTH")
    valid_phoneme_types = ["c", "C", "v", "V"]

    phoneme_types_user = list(phoneme_pattern) if isinstance(phoneme_pattern, str) else phoneme_pattern
    phoneme_types = list(filter(lambda p: p in valid_phoneme_types, phoneme_types_user))
    if phoneme_types_user != phoneme_types:
        print(f"Warning: ignoring invalid phoneme types {phoneme_types_user} -> {phoneme_types}. "
              f"Valid phoneme types in pattern: {valid_phoneme_types}")

    print(f"Search for phoneme-pattern '{''.join(phoneme_types)}'")

    phonemes = new_read_phonemes()

    multi_consonants = bin_feats.consonants
    single_consonants = list(filter(lambda x: len(x) == 1, multi_consonants))
    vowels = bin_feats.long_vowels
    long_vowels = list(filter(lambda x: len(x) == 2, bin_feats.long_vowels))

    #print(multi_consonants)
    #print(single_consonants)
    print(vowels)
    #print(long_vowels)

    exit()

    prefixes = tuple(consonants)
    postfixes = tuple(vowels)
    syll_length = prefix_length + postfix_length

    syllables = list(filter(
        lambda s: s.syll.startswith(prefixes) and s.syll.endswith(postfixes) and len(s.syll) == syll_length,
        syllables
    ))

    return syllables


if __name__ == '__main__':
    syllables = read_syllables()
    print(syllables[:10])

    # bin_feats = read_binary_features()
    # print(bin_feats)

    # phonemes = read_phonemes()
    # print(phonemes[:10])

    syllables = new_select_consonant_vowel(syllables)
    print(syllables[:10])
