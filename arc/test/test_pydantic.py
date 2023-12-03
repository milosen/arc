from pydantic import ValidationError

from arc.types import *

# lexica[words[syllables[phonemes]]]
example_word = {
    "id": "ka:fu:ry:",
    "features": [],
    "info": {},
    "syllables": [
        {
            "id": "ka:",
            "features": [],
            "custom_features": [],
            "info": {},
            "phonemes": [
                {
                    "id": "k",
                    "info": {},
                    "order": [],
                    "features": [],
                },
                {
                    "id": "a:",
                    "info": {},
                    "order": [],
                    "features": []
                }
            ]
        },
        {
            "id": "fu:",
            "features": [],
            "custom_features": [],
            "info": {},
            "phonemes": [
                {
                    "id": "f",
                    "info": {},
                    "order": [],
                    "features": []
                },
                {
                    "id": "u:",
                    "info": {},
                    "order": [],
                    "features": []
                }
            ]
        },
        {
            "id": "ry:",
            "info": {},
            "features": [],
            "custom_features": [],
            "phonemes": [
                {
                    "id": "r",
                    "info": {},
                    "order": [],
                    "features": []
                },
                {
                    "id": "y:",
                    "info": {},
                    "order": [],
                    "features": []
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
        list_syllables = ContainerARC()
        for syll in word:
            list_syllables.append(syll)

        print(list_syllables)
        print('fu:' in list_syllables)


if __name__ == '__main__':
    unittest.main()
