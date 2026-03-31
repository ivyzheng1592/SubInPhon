# 2025/06/05
# instances of languages together with their phoneme inventory and syllable structure

import json
import os
from Dataset.language_generator import BacknessHarmony, FinalDevoicing


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
        # Backward-compatibility alias: EnglishBH_full -> EnglishBH (if not explicitly defined)
        for name, lang in list(languages.items()):
            if name.endswith("_full"):
                base = name[:-5]
                if base not in languages:
                    languages[base] = lang
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


def _merge_variant(base, variant_spec):
    merged = dict(base)
    for key, value in variant_spec.items():
        merged[key] = value
    return merged


def _build_language(name, spec, variant=None):
    if variant:
        variants = spec.get("variants", {})
        if variant not in variants:
            raise ValueError(f"Unknown variant '{variant}' for language '{name}'")
        spec = _merge_variant(spec, variants[variant])

    onset = _build_onset(spec["onset"])
    coda = _build_coda(spec["coda"])
    vowel = _build_vowel(spec["vowel"])
    syll_struct = spec["syll_struct"]
    lang_name = spec.get("lang_name", name)
    allowed_templates = spec.get("allowed_templates")

    lang_type = spec["type"]
    if lang_type == "BacknessHarmony":
        return BacknessHarmony(onset, coda, vowel, syll_struct, lang_name,
                               allowed_templates=allowed_templates)
    if lang_type == "FinalDevoicing":
        return FinalDevoicing(onset, coda, vowel, syll_struct, lang_name)

    raise ValueError(f"Unknown language type: {lang_type}")


registry = LanguageRegistry()

# Backwards-compatible dict used by existing code
languages = registry.build()
