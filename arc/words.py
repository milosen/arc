import random
from collections import deque

from arc.functional import filter_iterable
from arc.syllables import *
from arc.types import *


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


def generate_trisyll_word(syllables, f_cls):
    allIdx = set([syllables.index(i) for i in syllables])
    while True:
        syl_1 = syllables.index(random.sample(syllables, 1)[0])
        set_1 = generate_subset_sylls(syl_1, allIdx, f_cls)
        if set_1:
            syl_2 = random.sample(list(set_1), 1)[0]
            set_2 = generate_subset_sylls(syl_2, allIdx, f_cls)
            if set_2:
                set_3 = set_1.intersection(set_1, set_2)
                if set_3:
                    syl_3 = random.sample(list(set_3), 1)[0]
                    break
    return syllables[syl_1] + syllables[syl_2] + syllables[syl_3]


# CHECK IF A TRIPLET HAS NON-ZERO (gram) OR UNIFORM (Gram) BIGRAM AND TRIGRAM LOG-PROBABILITY
def get_corpus_gram_stats_triplet(iTrip, Gram2, Gram3):
    """TODO: Change implementation for other syllables-settings?"""
    GoodT = True
    seg_1 = iTrip[1:4]
    seg_2 = iTrip[4:7]
    if any(i not in Gram2 for i in [seg_1, seg_2]):
        GoodT = False
    seg_1 = iTrip[0:4]
    seg_2 = iTrip[1:6]
    seg_3 = iTrip[3:7]
    seg_4 = iTrip[4:9]
    if any(i not in Gram3 for i in [seg_1, seg_2, seg_3, seg_4]):
        GoodT = False
    return GoodT


def has_valid_gram_stats(word: Word, valid_bigrams: NgramsList, valid_trigrams: NgramsList):
    """TODO: Change implementation for other syllables-settings?"""
    """Assuming that the individual syllables in the word are valid, we still have to check if the 
    bigrams and trigrams that have formed at the syllable transitions are valid.
    
    Example:
    word = "heː|pɛː|naː"
    valid_sylls = ["heː", "pɛː", "naː"]
    So, in addition, we have to check that the bigrams "eːp" and "ɛː|n" at the transitions are contained 
    in the valid bigrams list
    """
    word = "".join([s.syll for s in word])
    valid = True
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


def generate_nsyll_words(syllables: SyllablesList, f_cls, n_sylls=3, n_look_back=2, n_tries=1_000_000) -> List:
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
        word = tuple([syllables[syl_1], syllables[syl_2], syllables[syl_3]])
        words.add(word)
    return list(words)


def generate_words_2(syllables, f_cls, n_sylls=3, n_look_back=2, n_tries=1_000_000) -> List:
    words = set()
    all_idx = set(range(len(syllables)))
    idx_sets = deque([all_idx]*n_look_back)
    for _ in tqdm(range(n_tries)):
        word = ""
        for _ in range(n_sylls):
            if not idx_sets[-1]:
                break
            idx = random.choice(list(idx_sets[-1]))
            word += syllables[idx]
            new_idx_set = generate_subset_sylls(idx, all_idx, f_cls)
            idx_sets.append(new_idx_set.intersection(*idx_sets))
        words.add(word)
    return words


def filter_rare_onset_phonemes(syllables: SyllablesList, phonemes: PhonemesList,
                               p_threshold: float = 0.05) -> PhonemesList:
    print("FIND SYLLABLES THAT ARE RARE AT THE ONSET OF A WORD")
    candidate_onset_phonemes = set([syll.syll[0] for syll in syllables])

    onset_phonemes = [x.phon for x in phonemes if x.order == 1]
    all_phonemes = [x.phon for x in phonemes]

    rare_phonemes = set()
    for phon in tqdm(candidate_onset_phonemes):
        phoneme_prob = onset_phonemes.count(phon) / all_phonemes.count(phon)
        if phoneme_prob < p_threshold:
            rare_phonemes.add(phon)
    rare_phonemes = list(rare_phonemes)
    return rare_phonemes


def generate_words():
    print("LOAD SYLLABLES")

    with open(os.path.join(RESULTS_DEFAULT_PATH, "syllables.pickle"), 'rb') as f:
        syllables: SyllablesList = pickle.load(f)

    bin_feats = read_binary_features()
    print(bin_feats)

    bigrams = read_bigrams()
    print(bigrams[:10])

    trigrams = read_trigrams()
    print(trigrams[:10])

    print("SELECT BIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    bigrams_uniform = filter_iterable(function=lambda s: s.p_unif > 0.05, iterable=bigrams)

    print("SELECT TRIGRAMS WITH UNIFORM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    trigrams_uniform = filter_iterable(function=lambda s: s.p_unif > 0.05, iterable=trigrams)

    CVsyl = [s.syll for s in syllables]
    bigrams_list = [b.ngram for b in bigrams]
    trigrams_list = [t.ngram for t in trigrams]

    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)
    numbs = bin_feat.numbs
    phons = bin_feat.phons
    labls = bin_feat.labels

    print("COMPUTE MULTI-FEATURE COMBINATIONS FOR EACH CONSONANT (MANNER AND PLACE) AND VOWEL (IDENTITY)")
    i_Son = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('son')] == '+']
    i_Plo = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('son')] != '+'
             and numbs[phons.index(CVsyl[i][0])][labls.index('cont')] != '+']
    i_Fri = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('son')] != '+'
             and numbs[phons.index(CVsyl[i][0])][labls.index('cont')] == '+']
    i_Lab = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('lab')] == '+']
    i_Den = [i for i in range(len(CVsyl)) if numbs[phons.index(CVsyl[i][0])][labls.index('cor')] == '+'
             and numbs[phons.index(CVsyl[i][0])][labls.index('hi')] != '+']
    i_Oth = [i for i in range(len(CVsyl)) if i not in i_Lab and i not in i_Den]
    idx_A = [i for i in range(len(CVsyl)) if 'a' in CVsyl[i][1:]]
    idx_E = [i for i in range(len(CVsyl)) if 'e' in CVsyl[i][1:]]
    idx_I = [i for i in range(len(CVsyl)) if 'i' in CVsyl[i][1:]]
    idx_O = [i for i in range(len(CVsyl)) if 'o' in CVsyl[i][1:]]
    idx_U = [i for i in range(len(CVsyl)) if 'u' in CVsyl[i][1:]]
    idx_Ä = [i for i in range(len(CVsyl)) if 'ɛ' in CVsyl[i][1:]]
    idx_Ö = [i for i in range(len(CVsyl)) if 'ø' in CVsyl[i][1:]]
    idx_Ü = [i for i in range(len(CVsyl)) if 'y' in CVsyl[i][1:]]
    f_Mnr = [set(i_Son), set(i_Plo), set(i_Fri)]
    f_Plc = [set(i_Oth), set(i_Lab), set(i_Den)]
    f_Vow = [set(idx_A), set(idx_E), set(idx_I), set(idx_O), set(idx_U),
             set(idx_Ä), set(idx_Ö), set(idx_Ü)]
    f_Cls = [f_Mnr, f_Plc, f_Vow]

    REGENERATE_WORDS = True
    CHECK_VALID_GERMAN = False

    if not os.path.exists(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv')) or REGENERATE_WORDS:
        print("GENERATE LIST OF TRISYLLABIC WORDS WITH NO OVERLAP OF COMPLEX PHONETIC FEATURES ACROSS SYLLABLES")
        # words = list(set([generate_trisyll_word(CVsyl, f_Cls) for _ in tqdm(range(1_000_000))]))
        words = generate_nsyll_words(syllables, f_Cls)
        print("Words generated: ", len(words))

        # SAVE LIST OF TRIPLETS IN ONE CSV FILE
        with open(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv'), 'w') as f:
            w = csv.writer(f)
            for word in words:
                w.writerows([["".join([s.syll for s in word]), 0]])

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

    if CHECK_VALID_GERMAN:
        print("LOAD WORDS FROM CSV FILE AND SELECT THOSE THAT CANNOT BE MISTAKEN FOR GERMAN WORDS")
        fdata = list(csv.reader(open(os.path.join(RESULTS_DEFAULT_PATH, 'words.csv'), "r"), delimiter='\t'))
        rows = [row[0].split(",") for row in fdata]
        words_valid = [row[0] for row in rows if word[1] == "0"]
        words_valid_german = []
        for word in tqdm(words):
            if tuple([s.syll for s in word]) in words_valid:
                words_valid_german.append(word)
        words = words_valid_german

    print("EXCLUDE WORDS WITH LOW ONSET SYLLABLE PROBABILITY")
    phonemes = read_phonemes()
    rare_phonemes = filter_rare_onset_phonemes(syllables, phonemes)
    words = filter_iterable(lambda w: w[0].syll[0] not in rare_phonemes, words)

    print("SELECT WORDS WITH UNIFORM BIGRAM AND NON-ZERO TRIGRAM LOG-PROBABILITY OF OCCURRENCE IN THE CORPUS")
    # words_valid_german = [word for word in tqdm(words) if has_valid_gram_stats(word, bigrams_list, trigrams_list)]
    words_valid_german = list(filter(lambda word: has_valid_gram_stats(word, bigrams_list, trigrams_list), tqdm(words)))
    print(len(words_valid_german), len(words))

    print("SAVE WORDS")
    fname = os.path.join(RESULTS_DEFAULT_PATH, 'words.pickle')
    fdata = [words_valid_german, words]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    generate_words()
