from arc.stream import *

import unittest


class MyTestCase(unittest.TestCase):
    def test_1(self):
        with self.assertRaises(TypeError):
            merge_with_corpus(read_feature_syllables("ccV", return_as_dict=False))

    def test_2(self):
        feature_syllables = read_feature_syllables("ccV", return_as_dict=True)

        sylls = merge_with_corpus(feature_syllables)

        print(f"\n".join([f"{s.as_dict()}" for s in sylls]))

        sylls = list(filter(lambda s: s.info["p_unif"] > 0.05, sylls))

        native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)

        for syll in sylls:
            print(f"{syll} has only native phonemes: ", all([(phon.id in native_phonemes) for phon in syll.phonemes]))
            # print(f"{syll} has native phonemes: ", syll in list(native_sylls))

        sylls = list(filter(lambda syll: all([(phon.id in native_phonemes) for phon in syll.phonemes]), sylls))
        print(f"\n".join([f"{s.as_dict()}" for s in sylls]))


if __name__ == '__main__':
    unittest.main()
