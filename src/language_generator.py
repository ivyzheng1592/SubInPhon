# created 2024/09/15
# updated 2025/06/05
# class of language patterns
# with information of the phoneme inventory and syllable structure of a language
# together with functions to generate and decompose stimuli for this language

import csv
import itertools
import math
import os
import random
from openpyxl import Workbook
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
        # Store the base language name from the configuration.
        self.lang_name = lang_name
        self.syll_struct = syll_struct
        self.word_struct = word_struct
        self.onset = onset
        self.coda = coda
        self.vowel = vowel
        self.variants = {}
        # Store the focus inventory used by downstream analysis code.
        self.focus = {}
        # Store the registry entry name used in generated filenames.
        self.registry_name = lang_name
        # Store the cached syllable inventory used during generation and sampling.
        self._syllable_templates_cache = None

    # Generate the dataset files for one language configuration.
    # Use iter_stimuli when the full dataset is needed.
    # Use sample_stimuli when only a sampled subset is needed.
    def generate_stimuli(
        self,
        seed: Optional[int] = None,
        sample_proportion: float = 1.0,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        pass

    # Yield every valid harmony and disharmony row for the language.
    def iter_stimuli(
        self,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
    ):
        pass

    # Return a sampled subset of the harmony and disharmony rows for the language.
    def sample_stimuli(
        self,
        template_sample_sizes: Dict[str, int],
        seed: Optional[int] = None,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
    ):
        pass

    # Write the template-count workbook for the current language pattern.
    def write_template_count(
        self,
        report_file: str,
        template_word_counts: Dict[str, int],
        template_sample_sizes: Dict[str, int],
    ) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = os.path.splitext(os.path.basename(report_file))[0][:31]

        # Write the header row for the template-count table.
        sheet.append([
            "template",
            "total_word_count",
            "sample_count",
        ])

        # Write one summary row per template.
        for struct in self.word_struct:
            sheet.append([
                struct,
                template_word_counts[struct],
                template_sample_sizes[struct],
            ])

        # Save the workbook to disk.
        workbook.save(report_file)

    # Write a list of dataset rows to one CSV file.
    def write_stimuli(
        self,
        rows: List[Dict[str, str]],
        csv_file: str,
        variant: Optional[str] = None,
    ) -> None:
        # Build the CSV header for the requested output format.
        include_variant = variant is not None
        headers = ["ur_syll", "sr_syll", "ur_string", "sr_string"]
        if include_variant:
            headers += ["ur_var", "sr_var"]

        # Write the header row and the dataset rows.
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for row in rows:
                writer.writerow([row[field] for field in headers])

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

    # Generate one dataset bundle and return the harmony/disharmony file paths.
    def generate_stimuli(
        self,
        seed: Optional[int] = None,
        sample_proportion: float = 1.0,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        # Use the default data/ folder when no custom output folder is provided.
        if output_dir is None:
            output_dir = "data"

        # Build the output paths for the CSV files and the count report.
        os.makedirs(output_dir, exist_ok=True)
        root = "_".join(part for part in [self.registry_name, directionality, property] if part)
        harmony_file = os.path.join(output_dir, f"{root}_harmony.csv")
        disharmony_file = os.path.join(output_dir, f"{root}_disharmony.csv")
        report_file = os.path.join(output_dir, f"{root}_template_counts.xlsx")

        # Count the available words for each template under the requested property.
        template_word_counts = {
            struct: self.count_template_stimuli(struct, property=property)
            for struct in self.word_struct
        }

        # Build the planned sample size for each template.
        template_sample_sizes = (
            self._allocate_template_sample_sizes(template_word_counts, sample_proportion)
            if sample_proportion < 1
            else {struct: template_word_counts[struct] for struct in self.word_struct}
        )

        # Write the planned template counts before generating the dataset rows.
        self.write_template_count(
            report_file,
            template_word_counts=template_word_counts,
            template_sample_sizes=template_sample_sizes,
        )

        # Collect the sampled rows before writing the output CSV files.
        if sample_proportion < 1:
            generated_rows = self.sample_stimuli(
                seed=seed,
                template_sample_sizes=template_sample_sizes,
                property=property,
                directionality=directionality,
                variant=variant,
            )
        else:
            # Collect the full generated rows before writing the output CSV files.
            generated_rows = {"harmony": [], "disharmony": []}
            for condition, row in self.iter_stimuli(
                property=property,
                directionality=directionality,
                variant=variant,
            ):
                generated_rows[condition].append(row)

        # Write the generated rows to the output CSV files.
        self.write_stimuli(generated_rows["harmony"], harmony_file, variant=variant)
        self.write_stimuli(generated_rows["disharmony"], disharmony_file, variant=variant)

        return {"harmony": harmony_file, "disharmony": disharmony_file}

    # Count the number of rows available for one template under the requested property.
    def count_template_stimuli(
        self,
        struct: str,
        property: str = "",
    ) -> int:
        # Count how many syllables of each vowel type are available at each slot.
        syll_templates = self._build_syllable_templates()
        word_templates = struct.split("-")
        syll_counts_by_vowel_type = [
            {vowel_type: len(syllables) for vowel_type, syllables in syll_templates[template].items()}
            for template in word_templates
        ]

        # Enumerate the vowel-type assignments that can fill the current template.
        vowel_assignments = list(
            itertools.product(*(slot_counts.keys() for slot_counts in syll_counts_by_vowel_type))
        )

        # Count the rows that match the requested property.
        template_word_count = 0
        for vowel_assignment in vowel_assignments:
            # Skip assignments that repeat a vowel type across slots.
            if property == "nonidentical" and len(set(vowel_assignment)) < len(vowel_assignment):
                continue
            assignment_word_count = 1
            for slot_idx, vowel_type in enumerate(vowel_assignment):
                assignment_word_count *= syll_counts_by_vowel_type[slot_idx][vowel_type]
            template_word_count += assignment_word_count
        return template_word_count

    # Convert a global sample proportion into one paired sample count per template.
    @staticmethod
    def _allocate_template_sample_sizes(
        template_word_counts: Dict[str, int],
        sample_proportion: float,
    ) -> Dict[str, int]:
        # Initialize the per-template sample sizes.
        sample_sizes = {struct: 0 for struct in template_word_counts}

        # Assign the proportional ceiling of the requested sample size to each template.
        for struct, template_word_count in template_word_counts.items():
            sample_sizes[struct] = min(template_word_count, math.ceil(template_word_count * sample_proportion))
        return sample_sizes

    # Build the legal syllable inventory for each syllable template and vowel type.
    def _build_syllable_templates(self) -> Dict[str, Dict[Tuple[str, str], List[Tuple[str, str, str]]]]:
        if self._syllable_templates_cache is not None:
            return self._syllable_templates_cache

        # Collect all legal syllables for each template and vowel-type group.
        # vowel_class is the tense/lax class used to decide whether a vowel can appear in a template.
        # vowel_type is the (height, tense/lax) key used to group syllables for counting and sampling.
        # Example:
        # {
        #     "010": {("high", "tense"): [("", "i", ""), ("", "u", "")]},
        #     "011": {("high", "lax"): [("", "I", "p"), ("", "U", "t")]},
        # }
        syll_templates = {t: {} for t in self.syll_struct.keys()}
        for vowel_label, feats in self.vowel.items():
            vowel_class = feats[2]
            vowel_type = (feats[1], feats[2])
            for template_bits, template_class in self.syll_struct.items():
                if vowel_class != template_class:
                    continue
                onset_list = self.onset if template_bits[0] == "1" else [""]
                coda_list = self.coda if template_bits[2] == "1" else [""]
                syll_templates[template_bits].setdefault(vowel_type, []).extend(
                    (onset, vowel_label, coda) for onset, coda in itertools.product(onset_list, coda_list)
                )
        self._syllable_templates_cache = syll_templates
        return self._syllable_templates_cache

    # Return the vowel that matches the trigger on height and tense.
    def _match_vowel(self, trigger_vowel: str, agree: str) -> str:
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

    # Map a base vowel to the corresponding vowel in the selected variant inventory.
    def _map_variant_vowel(self, base_vowel: str, variant: Optional[str]) -> str:
        variant_vowel = self.variants.get(variant)
        if not variant_vowel or base_vowel not in self.vowel:
            return base_vowel
        b_height = self.vowel[base_vowel][1]
        b_tense = self.vowel[base_vowel][2]
        b_backness = self.vowel[base_vowel][0]
        for alt_vowel, feats in variant_vowel.items():
            if feats[0] == b_backness and feats[1] == b_height and feats[2] == b_tense:
                return alt_vowel
        return base_vowel

    # Build one dataset row from a UR/SR pair.
    def _make_row(
        self,
        struct: str,
        ur_tuple: Tuple[Tuple[str, str, str], ...],
        sr_tuple: List[Tuple[str, str, str]],
        variant: Optional[str] = None,
    ) -> Dict[str, str]:
        # Build the dotted UR and SR syllable strings.
        ur_syll = ".".join(f"{onset}{vowel}{coda}" for onset, vowel, coda in ur_tuple)
        sr_syll = ".".join(f"{onset}{vowel}{coda}" for onset, vowel, coda in sr_tuple)

        # Build the shared row fields used by all generated datasets.
        row = {
            "template": struct,
            "ur_syll": ur_syll,
            "sr_syll": sr_syll,
            "ur_string": ur_syll.replace(".", ""),
            "sr_string": sr_syll.replace(".", ""),
        }

        # Add the variant-specific string columns when a variant inventory is requested.
        if variant is not None:
            ur_var_syll = ".".join(
                f"{onset}{self._map_variant_vowel(vowel, variant)}{coda}"
                for onset, vowel, coda in ur_tuple
            )
            sr_var_syll = ".".join(
                f"{onset}{self._map_variant_vowel(vowel, variant)}{coda}"
                for onset, vowel, coda in sr_tuple
            )
            row["ur_var"] = ur_var_syll.replace(".", "")
            row["sr_var"] = sr_var_syll.replace(".", "")
        return row

    # Build the harmony and disharmony SR tuples for one UR tuple.
    def _match_sr_tuple(
        self,
        ur_tuple: Tuple[Tuple[str, str, str], ...],
        directionality: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        vowels_in_word = [parts[1] for parts in ur_tuple]
        if directionality == "l2r":
            vh_vowels = [vowels_in_word[0]] + [self._match_vowel(vowel, "vh") for vowel in vowels_in_word[1:]]
            dh_vowels = [vowels_in_word[0]] + [self._match_vowel(vowel, "dh") for vowel in vowels_in_word[1:]]
        else:
            vh_vowels = [self._match_vowel(vowel, "vh") for vowel in vowels_in_word[:-1]] + [vowels_in_word[-1]]
            dh_vowels = [self._match_vowel(vowel, "dh") for vowel in vowels_in_word[:-1]] + [vowels_in_word[-1]]
        vh_sr_tuple = [(parts[0], vh_vowels[i], parts[2]) for i, parts in enumerate(ur_tuple)]
        dh_sr_tuple = [(parts[0], dh_vowels[i], parts[2]) for i, parts in enumerate(ur_tuple)]
        return vh_sr_tuple, dh_sr_tuple

    # Yield all harmony and disharmony rows for one word template.
    def iter_template_stimuli(
        self,
        struct: str,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
    ):
        # Build the legal syllable choices for each slot in the current word template.
        syll_templates = self._build_syllable_templates()
        word_templates = struct.split("-")
        syll_lists = [
            [syllable for syllables in syll_templates[template].values() for syllable in syllables]
            for template in word_templates
        ]

        # Generate one UR tuple at a time and skip rows that fail the requested property filter.
        for ur_tuple in itertools.product(*syll_lists):
            vowel_types = [(self.vowel[vowel][1], self.vowel[vowel][2]) for _, vowel, _ in ur_tuple]
            if property == "nonidentical" and len(set(vowel_types)) != len(vowel_types):
                continue

            # Build the harmony and disharmony SR tuples for the current UR tuple.
            vh_sr_tuple, dh_sr_tuple = self._match_sr_tuple(ur_tuple, directionality)

            # Yield one harmony row and one disharmony row for the current UR tuple.
            yield "harmony", self._make_row(struct, ur_tuple, vh_sr_tuple, variant=variant)
            yield "disharmony", self._make_row(struct, ur_tuple, dh_sr_tuple, variant=variant)

    # Yield all generated rows across every word template in the language.
    def iter_stimuli(
        self,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
    ):
        for struct in self.word_struct:
            yield from self.iter_template_stimuli(
                struct,
                property=property,
                directionality=directionality,
                variant=variant,
            )

    # Sample concrete rows from one template.
    def sample_template_stimuli(
        self,
        struct: str,
        sample_size: int,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        # Set the random generator for this template sample.
        rng = random.Random(seed)

        # Build the grouped syllable choices for each slot in the current template.
        syll_templates = self._build_syllable_templates()
        word_templates = struct.split("-")
        sylls_by_vowel_type = [syll_templates[template] for template in word_templates]
        syll_counts_by_vowel_type = [
            {vowel_type: len(syllables) for vowel_type, syllables in syllables_by_vowel_type.items()}
            for syllables_by_vowel_type in sylls_by_vowel_type
        ]
        vowel_assignments = list(
            itertools.product(*(slot_counts.keys() for slot_counts in syll_counts_by_vowel_type))
        )

        # Build the word-index range covered by each vowel assignment in this template.
        assignment_ranges = []
        running_total = 0
        for vowel_assignment in vowel_assignments:
            # Skip assignments that repeat a vowel type across slots.
            if property == "nonidentical" and len(set(vowel_assignment)) < len(vowel_assignment):
                continue
            assignment_word_count = 1
            for slot_idx, vowel_type in enumerate(vowel_assignment):
                assignment_word_count *= syll_counts_by_vowel_type[slot_idx][vowel_type]
            if assignment_word_count == 0:
                continue
            assignment_ranges.append((running_total, running_total + assignment_word_count, vowel_assignment))
            running_total += assignment_word_count

        # Choose the paired UR samples for the current template.
        sampled_rows = {"harmony": [], "disharmony": []}
        sample_size = min(sample_size, running_total)
        sampled_indices = sorted(rng.sample(range(running_total), sample_size))

        # Decode each sampled index into one UR tuple and append both paired outputs.
        for sampled_index in sampled_indices:
            for assignment_start, assignment_end, vowel_assignment in assignment_ranges:
                if sampled_index >= assignment_end:
                    continue
                index_in_assignment = sampled_index - assignment_start
                parts = []
                for slot_idx in range(len(sylls_by_vowel_type) - 1, -1, -1):
                    syllable_list = sylls_by_vowel_type[slot_idx][vowel_assignment[slot_idx]]
                    choice_idx = index_in_assignment % len(syllable_list)
                    index_in_assignment //= len(syllable_list)
                    parts.append(syllable_list[choice_idx])
                ur_tuple = tuple(reversed(parts))
                vh_sr_tuple, dh_sr_tuple = self._match_sr_tuple(ur_tuple, directionality)
                sampled_rows["harmony"].append(self._make_row(struct, ur_tuple, vh_sr_tuple, variant=variant))
                sampled_rows["disharmony"].append(self._make_row(struct, ur_tuple, dh_sr_tuple, variant=variant))
                break

        return sampled_rows

    # Sample rows across all templates using per-template allocations.
    def sample_stimuli(
        self,
        template_sample_sizes: Dict[str, int],
        seed: Optional[int] = None,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        # Collect the sampled rows from every template into one output bundle.
        sampled_rows = {
            "harmony": [],
            "disharmony": [],
        }
        for struct_idx, struct in enumerate(self.word_struct):
            # Sample the requested number of paired rows from the current template.
            template_seed = None if seed is None else seed + struct_idx
            template_rows = self.sample_template_stimuli(
                struct,
                template_sample_sizes[struct],
                property=property,
                directionality=directionality,
                variant=variant,
                seed=template_seed,
            )

            # Append the current template's rows to the full sampled dataset.
            sampled_rows["harmony"].extend(template_rows["harmony"])
            sampled_rows["disharmony"].extend(template_rows["disharmony"])
        return sampled_rows

    # Decompose one tokenized word into its syllable parts.
    def decompose_stimuli(self, word_list: List[str]) -> List[List[Any]]:
        # Copy the input tokens before removing boundary markers.
        word_copy = word_list.copy()  # copy of word for token removal

        # Remove the leading SOS token.
        if word_copy and word_copy[0] == "<SOS>":
            word_copy.pop(0)
        # Remove trailing PAD tokens.
        while word_copy and word_copy[-1] == "<PAD>":
            word_copy.pop()
        # Remove trailing EOS tokens.
        while word_copy and word_copy[-1] == "<EOS>":
            word_copy.pop()

        sylls = []
        pos = 0

        # Parse the word from left to right into syllables.
        while pos < len(word_copy):
            syll = [None, None, None]

            # Read an optional onset consonant.
            if word_copy[pos] in self.onset:
                syll[0] = word_copy[pos]
                pos += 1

            # Read the vowel slot.
            if pos >= len(word_copy) or word_copy[pos] not in self.vowel:
                return [False, False, False]
            syll[1] = word_copy[pos]
            pos += 1

            # Read or validate the coda slot.
            v_class = self.vowel[syll[1]][2]
            if v_class == "lax":
                # Read the required coda after a lax vowel.
                if pos >= len(word_copy) or word_copy[pos] not in self.coda:
                    return [False, False, False]
                syll[2] = word_copy[pos]
                pos += 1
            else:
                # Reject a coda after a tense vowel.
                if pos < len(word_copy) and word_copy[pos] in self.coda and word_copy[pos] not in self.onset:
                    return [False, False, False]

            sylls.append(syll)

        # Build the template string from the parsed syllables.
        bits = []
        for o, v, c in sylls:
            bits.append(("1" if o else "0") + "1" + ("1" if c else "0"))
        if "-".join(bits) not in self.word_struct:
            return [False, False, False]

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

    def generate_stimuli(
        self,
        seed: Optional[int] = None,
        sample_proportion: float = 1.0,
        property: str = "",
        directionality: str = "l2r",
        variant: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        pass

    def decompose_stimuli(self, word_list: List[str]):
        pass
