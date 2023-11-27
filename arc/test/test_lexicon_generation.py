from arc.stream import *


if __name__ == '__main__':
    with open(os.path.join(RESULTS_DEFAULT_PATH, 'words_test.csv'), 'w') as f:
        writer = csv.writer(f)
        for word in words:
            writer.writerows([[word.id, 0]])

    print(len(words))

    bin_feat = read_binary_features(BINARY_FEATURES_DEFAULT_PATH)

    oscillation_patterns = [tuple([i for i in np.roll((1, 0, 0, 1, 0, 0), j)]) for j in range(N_SYLLABLES_PER_WORD)]

    print("EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR")
    features = list(map(lambda w: binary_feature_matrix(w, bin_feat), tqdm(words)))

    print("compute word overlap matrix")
    overlap = compute_word_overlap_matrix(words=words, features=features, oscillation_patterns=oscillation_patterns)

    lexicon_generator_1 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)
    lexicon_generator_2 = sample_min_overlap_lexicon(words, overlap, n_words=4, max_overlap=1, max_yields=1000)

    # print pairwise lexicon generation
    for lexicon_1, lexicon_2 in itertools.product(lexicon_generator_1, lexicon_generator_2):

        # check if the lexicons are compatible, i.e. they should not have repeating syllables
        all_sylls = [s.id for lexicon in [lexicon_1, lexicon_2] for word in lexicon.words for s in word.syllables]
        if not len(set(all_sylls)) == len(all_sylls):
            continue

        print("Found compatible lexicons: ", lexicon_1.id, lexicon_1.info["cumulative_overlap"], lexicon_2.id, lexicon_2.info["cumulative_overlap"])

        s1w = None
        for stream_1_word_randomized in sample_word_randomization(lexicon, randomized_word_indexes):
            s1w = check_rhythmicity(stream_1_word_randomized, oscillation_patterns, bin_feat)

            if s1w:
                break

        if s1w:
            print("Lexicon:", lexicon.id)
            print("Cumulative overlap of words in lexicon:", lexicon.info["cumulative_overlap"])

            stream, stream_info = s1w
            print("Stream: ", "".join(syllable for syllable in stream))
            print("Rhythmicity Indexes: ", stream_info["rhythmicity_indexes"])
            break