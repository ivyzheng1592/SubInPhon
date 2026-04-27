# 2025/06/05
# instances of languages together with their phoneme inventory and syllable structure

import json
import os
from typing import Any, Optional, Dict, List

from language_generator import BacknessHarmony, FinalDevoicing


class LanguageRegistry:
    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "languages_config.json")
        self._config_path = config_path
        self._specs = self._load_specs()

    def _load_specs(self) -> Dict[str, Any]:
        with open(self._config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def names(self) -> List[str]:
        return list(self._specs.keys())

    def build_one(self, name: str, variant: Optional[str] = None) -> Any:
        spec = self._specs[name]
        return _build_language(name, spec, variant=variant)

    def build(self, variant: Optional[str] = None) -> Dict[str, Any]:
        languages = {name: _build_language(name, spec, variant=variant) for name, spec in self._specs.items()}
        return languages


def _build_onset(onset_spec: Any) -> Dict[str, Any]:
    if isinstance(onset_spec, list):
        return {onset: None for onset in onset_spec}
    if isinstance(onset_spec, dict):
        onset = {}
        for label, phones in onset_spec.items():
            onset.update({phone: label for phone in phones})
        return onset
    raise ValueError("onset must be a list or a dict")


def _build_coda(coda_spec: Any) -> Dict[str, Any]:
    if isinstance(coda_spec, list):
        return {coda: None for coda in coda_spec}
    if isinstance(coda_spec, dict):
        coda = {}
        for label, phones in coda_spec.items():
            coda.update({phone: label for phone in phones})
        return coda
    raise ValueError("coda must be a list or a dict")


def _build_vowel(vowel_spec: Any) -> Dict[str, Any]:
    if isinstance(vowel_spec, list):
        return {vowel: None for vowel in vowel_spec}
    if isinstance(vowel_spec, dict):
        return vowel_spec
    raise ValueError("vowel must be a list or a dict")




def _template_to_bits(template: str) -> str:
    if template in ("010", "110", "011", "111"):
        return template
    mapping = {"V": "010", "CV": "110", "VC": "011", "CVC": "111"}
    if template in mapping:
        return mapping[template]
    raise ValueError(f"Unknown syllable template: {template}")


def _convert_syll_struct(syll_struct: Dict[str, Any]) -> Dict[str, Any]:
    return {_template_to_bits(k): v for k, v in syll_struct.items()}


def _convert_word_struct(word_struct: List[str]) -> List[str]:
    converted = []
    for struct in word_struct:
        parts = struct.split("-")
        converted.append("-".join(_template_to_bits(p) for p in parts))
    return converted


def _build_language(name: str, spec: Dict[str, Any], variant: Optional[str] = None) -> Any:
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
    # Keep the registry key separately from lang.lang_name so generated dataset
    # filenames can follow the selected experiment entry (e.g. EnglishBH_shortened).
    # Example: lang.lang_name == "EnglishBH", but lang.registry_name == "EnglishBH_shortened".
    lang.registry_name = name
    return lang


registry = LanguageRegistry()

languages = registry.build()

if __name__ == "__main__":
    import hyper_params as hp

    language = languages[hp.lang_name]
    variant = "aud_vowel" if "aud_vowel" in language.variants else None

    # Example: generate a standalone sampled dataset outside the experiment loop.
    # This writes harmony/disharmony CSVs plus a template-count Excel report to a custom folder.
    example_output_dir = os.path.join("results", "standalone_generation_example")
    sampled_files = language.generate_stimuli(
        seed=hp.base_seed,
        sample_proportion=hp.data_proportion,
        property=hp.property,
        directionality=hp.directionality[0],
        variant=variant,
        output_dir=example_output_dir,
    )
    print(f"Standalone sampled files written to {example_output_dir}: {sampled_files}")
