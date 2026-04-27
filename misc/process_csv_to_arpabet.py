#!/usr/bin/env python3
"""
Script to process EnglishBH_shortened_l2r_harmony.csv and generate a two-column text file:
- Column 1: ur_string
- Column 2: ur_var converted to ARPABET with spaces between phones
"""

import csv
import os

# IPA to ARPABET mapping based on common phonetic correspondences
IPA_TO_ = {
    # Vowels (standalone only - diphthongs handled separately)
    'i': 'IY',   # high front tense
    'u': 'UW',   # high back tense
    'ɪ': 'IH',   # high front lax
    'ɛ': 'EH',   # mid front lax
    'ʊ': 'UH',   # high back lax
    'ɔ': 'AO',   # mid back lax

    # Consonants
    'm': 'M',    # bilabial nasal
    'n': 'N',    # alveolar nasal
    'ŋ': 'NG',   # velar nasal
    'p': 'P',    # voiceless bilabial plosive
    't': 'T',    # voiceless alveolar plosive
    'k': 'K',    # voiceless velar plosive
    'b': 'B',    # voiced bilabial plosive
    'd': 'D',    # voiced alveolar plosive
    'g': 'G',    # voiced velar plosive
    'f': 'F',    # voiceless labiodental fricative
    's': 'S',    # voiceless alveolar fricative
    'v': 'V',    # voiced labiodental fricative
    'z': 'Z',    # voiced alveolar fricative
    'ʃ': 'SH',   # voiceless postalveolar fricative
    'ʒ': 'ZH',   # voiced postalveolar fricative
    'θ': 'TH',   # voiceless dental fricative
    'ð': 'DH',   # voiced dental fricative
    'h': 'HH',   # voiceless glottal fricative
}

# IPA to ARPABET mapping based on common phonetic correspondences
IPA_TO_ARPABET = {
    # Vowels (standalone only - diphthongs handled separately)
    'i': 'IY',   # high front tense
    'u': 'UW',   # high back tense
    'ɪ': 'IH',   # high front lax
    'ɛ': 'EH',   # mid front lax
    'ʊ': 'UH',   # high back lax
    'ɔ': 'AO',   # mid back lax

    # Consonants
    'm': 'M',    # bilabial nasal
    'n': 'N',    # alveolar nasal
    'ŋ': 'NG',   # velar nasal
    'p': 'P',    # voiceless bilabial plosive
    't': 'T',    # voiceless alveolar plosive
    'k': 'K',    # voiceless velar plosive
    'b': 'B',    # voiced bilabial plosive
    'd': 'D',    # voiced alveolar plosive
    'g': 'G',    # voiced velar plosive
    'f': 'F',    # voiceless labiodental fricative
    's': 'S',    # voiceless alveolar fricative
    'v': 'V',    # voiced labiodental fricative
    'z': 'Z',    # voiced alveolar fricative
    'ʃ': 'SH',   # voiceless postalveolar fricative
    'ʒ': 'ZH',   # voiced postalveolar fricative
    'θ': 'TH',   # voiceless dental fricative
    'ð': 'DH',   # voiced dental fricative
    'h': 'HH',   # voiceless glottal fricative
}

# Diphthong mappings (processed before individual characters)
DIPHTHONGS = {
    'eɪ': 'EY',  # mid front tense + high front lax -> EY
    'oʊ': 'OW',  # mid back tense + high back lax -> OW
}


def ipa_to_arpabet(ipa_string):
    """
    Convert IPA phonetic string to ARPABET format with spaces between phones.
    Handles diphthongs first, then individual characters.

    Args:
        ipa_string (str): String in IPA notation

    Returns:
        str: ARPABET string with spaces between phones
    """
    # First, replace diphthongs with special markers
    working_string = ipa_string
    for diphthong, arpabet in DIPHTHONGS.items():
        working_string = working_string.replace(diphthong, f'[{arpabet}]')

    # Then process remaining individual characters
    arpabet_phones = []
    i = 0
    while i < len(working_string):
        # Check if we have a diphthong marker
        if working_string[i] == '[':
            # Find the closing bracket
            end_idx = working_string.find(']', i)
            if end_idx != -1:
                # Extract the ARPABET code
                arpabet_code = working_string[i+1:end_idx]
                arpabet_phones.append(arpabet_code)
                i = end_idx + 1
                continue

        # Process individual character
        char = working_string[i]
        if char in IPA_TO_ARPABET:
            arpabet_phones.append(IPA_TO_ARPABET[char])
        else:
            # If character not in mapping, keep as is (for debugging)
            arpabet_phones.append(char)
        i += 1

    return ' '.join(arpabet_phones)


def main():
    # File paths
    csv_file = os.path.join('..', 'Dataset', 'EnglishBH_shortened_l2r_harmony.csv')
    output_file = 'englishbh_arpabet.txt'

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return

    print(f"Reading from: {csv_file}")
    print(f"Writing to: {output_file}")

    with open(csv_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)

        for row in reader:
            ur_string = row['ur_string']
            ur_var = row['ur_var']

            # Column 1: ur_string (not duplicated)
            col1 = ur_string

            # Column 2: ur_var converted to ARPABET with spaces
            col2 = ipa_to_arpabet(ur_var)

            # Write tab-separated line
            outfile.write(f"{col1}\t{col2}\n")

    print(f"Processing complete. Output written to {output_file}")
    print(f"Total lines processed: {sum(1 for _ in open(csv_file)) - 1}")  # -1 for header


if __name__ == "__main__":
    main()