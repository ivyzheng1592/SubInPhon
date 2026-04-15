# 2025/06/05
# instances of languages together with their phoneme inventory and syllable structure

import json
import os
from language_generator import BacknessHarmony, FinalDevoicing


class LanguageRegistry:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "languages_config.json")
        self._config_path = config_path
        self._specs = self._load_specs()

    def _load_specs(self):
        with open(self._config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def names(self):
        return list(self._specs.keys())

    def build_one(self, name, variant=None):
        spec = self._specs[name]
        return _build_language(name, spec, variant=variant)

    def build(self, variant=None):
        languages = {name: _build_language(name, spec, variant=variant) for name, spec in self._specs.items()}
        return languages


def _build_onset(onset_spec):
    if isinstance(onset_spec, list):
        return {onset: None for onset in onset_spec}
    if isinstance(onset_spec, dict):
        onset = {}
        for label, phones in onset_spec.items():
            onset.update({phone: label for phone in phones})
        return onset
    raise ValueError("onset must be a list or a dict")


def _build_coda(coda_spec):
    if isinstance(coda_spec, list):
        return {coda: None for coda in coda_spec}
    if isinstance(coda_spec, dict):
        coda = {}
        for label, phones in coda_spec.items():
            coda.update({phone: label for phone in phones})
        return coda
    raise ValueError("coda must be a list or a dict")


def _build_vowel(vowel_spec):
    if isinstance(vowel_spec, list):
        return {vowel: None for vowel in vowel_spec}
    if isinstance(vowel_spec, dict):
        return vowel_spec
    raise ValueError("vowel must be a list or a dict")




def _template_to_bits(template):
    if template in ("010", "110", "011", "111"):
        return template
    mapping = {"V": "010", "CV": "110", "VC": "011", "CVC": "111"}
    if template in mapping:
        return mapping[template]
    raise ValueError(f"Unknown syllable template: {template}")


def _convert_syll_struct(syll_struct):
    return {_template_to_bits(k): v for k, v in syll_struct.items()}


def _convert_word_struct(word_struct):
    converted = []
    for struct in word_struct:
        parts = struct.split("-")
        converted.append("-".join(_template_to_bits(p) for p in parts))
    return converted


def _build_language(name, spec, variant=None):
    onset = _build_onset(spec["onset"])
    coda = _build_coda(spec["coda"])
    vowel = _build_vowel(spec["vowel"])
    syll_struct = _convert_syll_struct(spec["syll_struct"])
    word_struct = _convert_word_struct(spec["word_struct"])
    lang_name = spec.get("lang_name", name)
    lang_type = spec["type"]
    if lang_type == "BacknessHarmony":
        lang = BacknessHarmony(onset, coda, vowel, syll_struct, word_struct, lang_name)
    elif lang_type == "FinalDevoicing":
        lang = FinalDevoicing(onset, coda, vowel, syll_struct, word_struct, lang_name)
    else:
        raise ValueError(f"Unknown language type: {lang_type}")

    lang.variants = spec.get("variants", {})
    return lang


registry = LanguageRegistry()

languages = registry.build()

if __name__ == "__main__":
    """
    languages["EnglishBH"].generate_stimuli(property="nonidentical", 
                                            directionality="l2r",
                                            variant="aud_vowel")
    languages["EnglishBH"].generate_stimuli(property="nonidentical", 
                                            directionality="r2l",
                                            variant="aud_vowel")
    languages["EnglishBH"].generate_stimuli(property="full",
                                            directionality="l2r",
                                            variant="aud_vowel")
    languages["EnglishBH"].generate_stimuli(property="full",
                                            directionality="r2l",
                                            variant="aud_vowel")
    """
    languages["EnglishBH_shortened"].generate_stimuli(property="shortened",
                                                      directionality="l2r")
    languages["EnglishBH_shortened"].generate_stimuli(property="shortened",
                                                      directionality="r2l")
