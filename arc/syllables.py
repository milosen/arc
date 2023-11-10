import pickle
from arc.io import *
from arc.types import *


def select_consonant_vowel(
        syllables: SyllablesList,
        bin_feats: BinaryFeatures,
        prefix_mode: bool = "single_consonant",
        postfix_mode: str = "long_vowel"
) -> SyllablesList:
    """Only select those syllables that follow the pattern:
    (single/multiple) consonant(s) followed by a (long) vowel"""
    print("SELECT CONSONANT-VOWEL SYLLABLES WITH LONG VOWEL LENGTH")

    if prefix_mode == "single_consonant":
        prefix_length = 1
        consonants = list(filter(lambda x: len(x) == prefix_length, bin_feats.consonants))
    elif prefix_mode == "multi_consonant":
        raise NotImplementedError(f"Multiple onset consonants not supported yet.")
    else:
        raise ValueError("Unknown prefix mode.")

    # TODO: implement other vowel modes
    if postfix_mode == "long_vowel":
        postfix_length = 2
        vowels = list(filter(lambda x: len(x) == postfix_length, bin_feats.long_vowels))
    elif postfix_mode in ["short_vowel", "vowel"]:
        raise NotImplementedError(f"Postfix mode {postfix_mode} not supported yet.")
    else:
        raise ValueError("Unknown postfix mode.")

    prefixes = tuple(consonants)
    postfixes = tuple(vowels)
    syll_length = prefix_length + postfix_length

    syllables = list(filter(
        lambda s: s.syll.startswith(prefixes) and s.syll.endswith(postfixes) and len(s.syll) == syll_length,
        syllables
    ))

    return syllables


def select_occurrence_probability(syllables: SyllablesList, p_threshold: float = 0.05) -> SyllablesList:
    print("SELECT CV SYLLABLES WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")

    syllables = list(
        filter(
            lambda s: s.p_unif > p_threshold,
            syllables
        )
    )

    return syllables


def select_native_phonemes(syllables: SyllablesList, phonemes: PhonemesList) -> SyllablesList:
    print("REMOVE SYLLABLES WITH NON-NATIVE PHONEMES")

    prefixes = tuple(phoneme.phon for phoneme in phonemes)

    syllables = list(
        filter(
            lambda s: s.syll.startswith(prefixes),
            syllables
        )
    )

    return syllables


def export_speech_synthesiser(syllables: SyllablesList):
    print("SAVE EACH SYLLABLE TO A TEXT FILE FOR THE SPEECH SYNTHESIZER")
    syllables_dir = os.path.join(RESULTS_DEFAULT_PATH, "syllables")
    os.makedirs(syllables_dir, exist_ok=True)
    c = [i.syll[0] for i in syllables]
    v = [i.syll[1] for i in syllables]
    c = ' '.join(c).replace('ʃ', 'sch').replace('ɡ', 'g').replace('ç', 'ch').replace('ʒ', 'dsch').split()
    v = ' '.join(v).replace('ɛ', 'ä').replace('ø', 'ö').replace('y', 'ü').split()
    t = [c[i] + v[i] for i in range(len(syllables))]
    for syllable, text in zip(syllables, t):
        synth_string = '<phoneme alphabet="ipa" ph=' + '"' + syllable.syll + '"' + '>' + text + '</phoneme>'
        with open(os.path.join(syllables_dir, f'{str(syllable.syll[0:2])}.txt'), 'w') as f:
            f.write(synth_string + "\n")
            csv.writer(f)


def generate_syllables():
    syllables = read_syllables()
    print(syllables[:10])

    bin_feats = read_binary_features()
    print(bin_feats)

    phonemes = read_phonemes()
    print(phonemes[:10])

    syllables = select_consonant_vowel(syllables, bin_feats)
    print(syllables[:10])

    syllables = select_occurrence_probability(syllables)
    print(syllables[:10])

    syllables = select_native_phonemes(syllables, phonemes)
    print(syllables[:10])

    print("SAVE SYLLABLES")
    with open(os.path.join(RESULTS_DEFAULT_PATH, "syllables.pickle"), 'wb') as f:
        pickle.dump(syllables, f, pickle.HIGHEST_PROTOCOL)

    export_speech_synthesiser(syllables)


if __name__ == '__main__':
    generate_syllables()
