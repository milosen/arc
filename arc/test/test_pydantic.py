from pydantic import ValidationError

from arc.types import *

# lexica[words[syllables[phonemes]]]
example_word = {
    "id": "ka:fu:ry:",
    "binary_features": [],
    "info": {},
    "syllables": [
        {
            "id": "ka:",
            "binary_features": [],
            "phonotactic_features": [],
            "info": {},
            "phonemes": [
                {
                    "id": "k",
                    "info": {},
                    "order": [],
                    "binary_features": [],
                },
                {
                    "id": "a:",
                    "info": {},
                    "order": [],
                    "binary_features": []
                }
            ]
        },
        {
            "id": "fu:",
            "binary_features": [],
            "phonotactic_features": [],
            "info": {},
            "phonemes": [
                {
                    "id": "f",
                    "info": {},
                    "order": [],
                    "binary_features": []
                },
                {
                    "id": "u:",
                    "info": {},
                    "order": [],
                    "binary_features": []
                }
            ]
        },
        {
            "id": "ry:",
            "info": {},
            "binary_features": [],
            "phonotactic_features": [],
            "phonemes": [
                {
                    "id": "r",
                    "info": {},
                    "order": [],
                    "binary_features": []
                },
                {
                    "id": "y:",
                    "info": {},
                    "order": [],
                    "binary_features": []
                }
            ]
        },
    ]
}

from arc.stream import *

import unittest


class MyTestCase(unittest.TestCase):
    def test_1(self):
        with self.assertRaises(ValidationError):
            Word(id="ka:fu:ry:", wrong_key="This should throw an exception.")

    def test_2(self):
        try:
            word = Word(**example_word)
            print(word.as_dict())
            lex = Lexicon(**{"id": "ka:fu:ry:ka:fu:ry:", "words": [example_word, example_word], "info": {}})
            print(lex.as_dict())
        except ValidationError as e:
            print(e.errors())

    def test_3(self):
        word = Word(**example_word)
        list_syllables = Register()
        for syll in word:
            list_syllables.append(syll)

        print(list_syllables)
        print('fu:' in list_syllables)


if __name__ == '__main__':
    unittest.main()
