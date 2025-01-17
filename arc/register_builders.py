from collections import OrderedDict
from arc.io import BINARY_FEATURES_DEFAULT_PATH
from arc.types.elements import Phoneme, Syllable, Word, Stream
from arc.types.registers import PhonemeRegister, SyllableRegister, WordRegister, StreamRegister
import logging

logger = logging.getLogger(__name__)

class RegisterBuilder:
    def __init__(self, register_type):
        self.register = register_type()

    def add_item(self, key: str, item):
        self.register[key] = item
        return self

    def build(self):
        return self.register

class PhonemeRegisterBuilder(RegisterBuilder):
    def __init__(self):
        super().__init__(PhonemeRegister)

    def add_item(self, key: str, item: Phoneme):
        # Custom strategy for adding phonemes
        self.register[key] = item
        return self
    
    def read_phonemes_csv(self, binary_features_path: str = BINARY_FEATURES_DEFAULT_PATH) -> Register:
        logger.info("READ MATRIX OF BINARY FEATURES FOR ALL IPA PHONEMES")

        with open(binary_features_path, "r", encoding='utf-8') as csv_file:
            fdata = list(csv.reader(csv_file))

        phons = [row[0] for row in fdata[1:]]
        feats = [row[1:] for row in fdata[1:]]
        phoneme_feature_labels = fdata[0][1:]

        assert phoneme_feature_labels == PHONEME_FEATURE_LABELS

        for phon, features in zip(phons, feats):
            if phon not in self.register or features == self.register[phon].info["features"]:
                self.add_item(Phoneme(id=phon, info={"features": features}))
            else:
                logger.info(
                    f"Phoneme '{phon}' with conflicting "
                    f"feature entries {features} != {self.register[phon].info['features']}.")
                # del phonemes_dict[phon]
    
    def validate_language(self):
        
        self.register = self.register.intersection(read_phoneme_corpus())

        return phonemes

class SyllableRegisterBuilder(RegisterBuilder):
    def __init__(self):
        super().__init__(SyllableRegister)

    def add_item(self, key: str, item: Syllable):
        if key not in self.register or self.register[key].info != item.info:
            self.register[key] = item
        else:
            logger.info(
                f"Syllable '{key}' with conflicting stats {item.info} != {self.register[key].info}."
            )
        return self

class WordRegisterBuilder(RegisterBuilder):
    def __init__(self):
        super().__init__(WordRegister)

    def add_item(self, key: str, item: Word):
        # Custom strategy for adding words
        self.register[key] = item
        return self

class StreamRegisterBuilder(RegisterBuilder):
    def __init__(self):
        super().__init__(StreamRegister)

    def add_item(self, key: str, item: Stream):
        # Custom strategy for adding streams
        self.register[key] = item
        return self