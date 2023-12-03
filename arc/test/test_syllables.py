from arc.functional import *
from arc.types import *

import unittest


class MyTestCase(unittest.TestCase):
    def test_1(self):
        with self.assertRaises(TypeError):
            merge_with_corpus(read_feature_syllables("cV", return_as_dict=False))

    def test_2(self):
        feature_syllables: List[Syllable] = read_feature_syllables("cV", return_as_dict=True)

        sylls = merge_with_corpus(feature_syllables)

        sylls = filter_uniform_syllables(sylls)

        native_phonemes = read_ipa_seg_order_of_phonemes(return_as_dict=True)
        sylls = filter_common_phoneme_syllables(sylls, native_phonemes)

        export_speech_synthesiser(sylls)


if __name__ == '__main__':
    unittest.main()
