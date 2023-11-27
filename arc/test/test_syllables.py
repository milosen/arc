from arc.stream import *

import unittest


class MyTestCase(unittest.TestCase):
    def test_1(self):
        with self.assertRaises(TypeError):
            merge_with_corpus(read_feature_syllables("ccV", return_as_dict=False))

    def test_2(self):
        feature_syllables = read_feature_syllables("cV", return_as_dict=True)

        sylls = merge_with_corpus(feature_syllables)

        print(f"\n".join([f"{s.as_dict()}" for s in sylls]))

        freqs = [s.info["freq"] for s in sylls]

        p_vals_uniform = stats.uniform.sf(abs(stats.zscore(np.log(freqs))))

        sylls = [s[0] for s in filter(lambda s: s[1] > 0.05, zip(sylls, p_vals_uniform))]

        native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)

        sylls = list(filter(lambda syll: all([(phon.id in native_phonemes) for phon in syll.phonemes]), sylls))

        print(f"\n".join([f"{s.as_dict()}" for s in sylls]))


if __name__ == '__main__':
    unittest.main()
