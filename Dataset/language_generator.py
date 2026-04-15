# created 2024/09/15
# updated 2025/06/05
# class of language patterns
# with information of the phoneme inventory and syllable structure of a language
# together with functions to generate and decompose stimuli for this language

import csv
import itertools
import os
from typing import Any, Optional, Dict, List, Tuple


class LanguagePattern:
    def __init__(
        self,
        onset: Dict[str, Any],
        coda: Dict[str, Any],
        vowel: Dict[str, Any],
        syll_struct: Dict[str, Any],
        word_struct: List[str],
        lang_name: str,
    ) -> None:
        # lang_name is the internal/base language label from the config.
        # The registry key used for dataset file naming is attached later as registry_name.
        # Example: for the "EnglishBH_shortened" entry, lang_name may still be "EnglishBH".
        self.lang_name = lang_name
        self.syll_struct = syll_struct
        self.word_struct = word_struct
        self.onset = onset
        self.coda = coda
        self.vowel = vowel
        self.variants = {}
        self.focus = {}  # different focus for different language pattern
        self.registry_name = lang_name

    def generate_stimuli(self, property: str = "", variant: Optional[str] = None):
        pass

    def decompose_stimuli(self, word_list: List[str]):
        pass


class BacknessHarmony(LanguagePattern):
    def __init__(
        self,
        onset: Dict[str, Any],
        coda: Dict[str, Any],
        vowel: Dict[str, Any],
        syll_struct: Dict[str, Any],
        word_struct: List[str],
        lang_name: str,
    ) -> None:
        super().__init__(onset, coda, vowel, syll_struct, word_struct, lang_name)
        self.focus = self.vowel

    # function to generate vowel harmony stimuli with specified phoneme inventory and syllable structure
    def generate_stimuli(
        self,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        variant_vowel = self.variants.get(variant)

        print(" - Generating stimuli:")
        def build_syll(parts: Tuple[str, str, str]) -> str:
            o, v, c = parts
            return f"{o}{v}{c}"

        def match_vowel(trigger_vowel: str, agree: str) -> str:
            t_height = self.vowel[trigger_vowel][1]
            t_tense = self.vowel[trigger_vowel][2]
            t_backness = self.vowel[trigger_vowel][0]
            for target_vowel, feats in self.vowel.items():
                if agree == "vh":
                    is_match = feats[0] == t_backness and feats[1] == t_height and feats[2] == t_tense
                else:
                    is_match = feats[0] != t_backness and feats[1] == t_height and feats[2] == t_tense
                if is_match:
                    return target_vowel
            raise RuntimeError(f"No vowel matches {agree} for {trigger_vowel}")

        def map_variant_vowel(base_vowel: str) -> str:
            # map base vowel to its variant counterpart with the same features
            if not variant_vowel:
                return base_vowel
            if base_vowel not in self.vowel:
                return base_vowel
            b_height = self.vowel[base_vowel][1]
            b_tense = self.vowel[base_vowel][2]
            b_backness = self.vowel[base_vowel][0]
            for v2, feats in variant_vowel.items():
                if feats[0] == b_backness and feats[1] == b_height and feats[2] == b_tense:
                    return v2
            return base_vowel

        # build syllable inventories per template (C/V slots tracked)
        syll_templates = {t: [] for t in self.syll_struct.keys()}
        for v, feats in self.vowel.items():
            v_class = feats[2]
            for template_bits, template_class in self.syll_struct.items():
                if v_class != template_class:
                    continue
                onset_list = self.onset if template_bits[0] == "1" else [""]
                coda_list = self.coda if template_bits[2] == "1" else [""]
                syll_templates[template_bits].extend(
                    (o, v, c) for o, c in itertools.product(onset_list, coda_list)
                )

        vh_list = []
        dh_list = []
        for struct in self.word_struct:
            # expand each word template into all UR/SR pairs
            word_templates = struct.split("-")
            # collect syllables for each slot
            syll_lists = [syll_templates[t] for t in word_templates]

            for parts_tuple in itertools.product(*syll_lists):
                # build UR word
                ur_syll = ".".join(build_syll(parts) for parts in parts_tuple)
                ur_string = ur_syll.replace(".", "")

                # extract vowel sequence for harmony
                vowels_in_word = [parts[1] for parts in parts_tuple]
                if directionality == "l2r":
                    # rightward harmony: match all later vowels to the first vowel
                    vh_vowels = [vowels_in_word[0]] + [
                        match_vowel(v, "vh") for v in vowels_in_word[1:]
                    ]
                    dh_vowels = [vowels_in_word[0]] + [
                        match_vowel(v, "dh") for v in vowels_in_word[1:]
                    ]
                else:  # r2l
                    # leftward harmony: match all earlier vowels to the last vowel
                    vh_vowels = [
                        match_vowel(v, "vh") for v in vowels_in_word[:-1]
                    ] + [vowels_in_word[-1]]
                    dh_vowels = [
                        match_vowel(v, "dh") for v in vowels_in_word[:-1]
                    ] + [vowels_in_word[-1]]

                if property == "nonidentical":
                    # skip if any identical vowels appear in the harmonized form
                    if len(set(vh_vowels)) < len(vh_vowels):
                        continue

                # rebuild SR word with harmonized vowels
                vh_parts = [
                    (parts[0], vh_vowels[i], parts[2]) for i, parts in enumerate(parts_tuple)
                ]
                vh_sr_syll = ".".join(build_syll(parts) for parts in vh_parts)
                vh_sr_string = vh_sr_syll.replace(".", "")

                if variant is not None:
                    # create variant forms by swapping only vowels
                    ur_var_syll = ".".join(f"{o}{map_variant_vowel(v)}{c}" for o, v, c in parts_tuple)
                    ur_var = ur_var_syll.replace(".", "")
                    vh_var_syll = ".".join(f"{o}{map_variant_vowel(v)}{c}" for o, v, c in vh_parts)
                    vh_var = vh_var_syll.replace(".", "")
                    vh_list.append([ur_syll, vh_sr_syll, ur_string, vh_sr_string, ur_var, vh_var])
                else:
                    vh_list.append([ur_syll, vh_sr_syll, ur_string, vh_sr_string])

                # rebuild SR word with disharmonized vowels
                dh_parts = [
                    (parts[0], dh_vowels[i], parts[2]) for i, parts in enumerate(parts_tuple)
                ]
                dh_sr_syll = ".".join(build_syll(parts) for parts in dh_parts)
                dh_sr_string = dh_sr_syll.replace(".", "")
                
                if variant is not None:
                    # variant disharmony uses the same vowel mapping
                    dh_var_syll = ".".join(f"{o}{map_variant_vowel(v)}{c}" for o, v, c in dh_parts)
                    dh_var = dh_var_syll.replace(".", "")
                    dh_list.append([ur_syll, dh_sr_syll, ur_string, dh_sr_string, ur_var, dh_var])
                else:
                    dh_list.append([ur_syll, dh_sr_syll, ur_string, dh_sr_string])

            print(f"Now generating syllable structure {struct}, accumulating to {len(vh_list)} pairs")

        print(" - Writing to file:")
        file_prefix = "_".join(
            part for part in [self.registry_name, directionality, property] if part
        )
        harmony_file = os.path.join("Dataset", f"{file_prefix}_harmony.csv")
        disharmony_file = os.path.join("Dataset", f"{file_prefix}_disharmony.csv")

        with open(harmony_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header
            header = ['ur_syll', 'sr_syll', 'ur_string', 'sr_string']
            if variant is not None:
                header += ['ur_var', 'sr_var']
            writer.writerow(header)
            # write stimuli list
            writer.writerows(vh_list)
        print("Harmony file ready.")

        with open(disharmony_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header
            header = ['ur_syll', 'sr_syll', 'ur_string', 'sr_string']
            if variant is not None:
                header += ['ur_var', 'sr_var']
            writer.writerow(header)
            # write stimuli list
            writer.writerows(dh_list)
        print("Disharmony file ready.")

        return vh_list, dh_list

    # function to decompose vowel harmony stimuli with specified phoneme inventory
    def decompose_stimuli(self, word_list: List[str]) -> List[List[Any]]:
        word_copy = word_list.copy()  # copy of word for token removal

        # remove SOS token in word
        if word_copy and word_copy[0] == "<SOS>":
            word_copy.pop(0)
        # remove PAD token in word
        while word_copy and word_copy[-1] == "<PAD>":
            word_copy.pop()
        # remove EOS token in word
        while word_copy and word_copy[-1] == "<EOS>":
            word_copy.pop()

        sylls = []
        pos = 0

        # parse left-to-right using tense/lax to decide open vs closed syllables
        while pos < len(word_copy):
            syll = [None, None, None]

            # optional onset
            if word_copy[pos] in self.onset:
                syll[0] = word_copy[pos]
                pos += 1

            # vowel is required for a syllable
            if pos >= len(word_copy) or word_copy[pos] not in self.vowel:
                return [[False, False, False]]
            syll[1] = word_copy[pos]
            pos += 1

            # decide coda vs next onset based on vowel class
            v_class = self.vowel[syll[1]][2]
            if v_class == "lax":
                # lax vowels must be closed
                if pos >= len(word_copy) or word_copy[pos] not in self.coda:
                    return [[False, False, False]]
                syll[2] = word_copy[pos]
                pos += 1
            else:
                # tense vowels must be open
                if pos < len(word_copy) and word_copy[pos] in self.coda and word_copy[pos] not in self.onset:
                    return [[False, False, False]]

            sylls.append(syll)

        # build template bits from parsed syllables and validate against word_struct
        bits = []
        for o, v, c in sylls:
            bits.append(("1" if o else "0") + "1" + ("1" if c else "0"))
        if "-".join(bits) not in self.word_struct:
            return [[False, False, False]]

        return sylls


class FinalDevoicing(LanguagePattern):
    def __init__(
        self,
        onset: Dict[str, Any],
        coda: Dict[str, Any],
        vowel: Dict[str, Any],
        syll_struct: Dict[str, Any],
        word_struct: List[str],
        lang_name: str,
    ) -> None:
        super().__init__(onset, coda, vowel, syll_struct, word_struct, lang_name)
        self.focus = self.coda

    # function to generate final devoicing stimuli with specified phoneme inventory and syllable structure
    def generate_stimuli(self, property: str = "", variant: Optional[str] = None):
        pass

    # function to decompose final devoicing stimuli with specified phoneme inventory
    def decompose_stimuli(self, word_list: List[str]):
        pass
