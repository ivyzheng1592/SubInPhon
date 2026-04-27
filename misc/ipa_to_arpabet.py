#!/usr/bin/env python3
"""
Script to process EnglishBH_shortened_l2r_harmony.csv and generate a two-column text file:
- Column 1: ur_string converted to English letters
- Column 2: ur_var converted to ARPABET with spaces between phones
"""

import csv
import os

# IPA to ARPABET mapping based on common phonetic correspondences
IPA_TO_ARPABET = {
    # Vowels (standalone only - diphthongs handled separately)
    'i': 'IY1',   # high front tense
    'u': 'UW1',   # high back tense
    'ɪ': 'IH1',   # high front lax
    'ɛ': 'EH1',   # mid front lax
    'ʊ': 'UH1',   # high back lax
    'ɔ': 'AO1',   # mid back lax

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

# IPA to English letters mapping for ur_string
IPA_TO_ENGLISH = {
    # Vowels (doubled to distinguish from consonants)
    'i': 'ii',   # high front tense
    'u': 'uu',   # high back tense
    'e': 'ee',   # mid front tense
    'o': 'oo',   # mid back tense
    'ɪ': 'i',   # high front lax
    'ɛ': 'e',   # mid front lax
    'ʊ': 'u',   # high back lax
    'ɔ': 'o',   # mid back lax

    # Consonants
    'm': 'm',    # bilabial nasal
    'n': 'n',    # alveolar nasal
    'ŋ': 'ng',   # velar nasal
    'p': 'p',    # voiceless bilabial plosive
    't': 't',    # voiceless alveolar plosive
    'k': 'k',    # voiceless velar plosive
    'b': 'b',    # voiced bilabial plosive
    'd': 'd',    # voiced alveolar plosive
    'g': 'g',    # voiced velar plosive
    'f': 'f',    # voiceless labiodental fricative
    's': 's',    # voiceless alveolar fricative
    'v': 'v',    # voiced labiodental fricative
    'z': 'z',    # voiced alveolar fricative
    'ʃ': 'sh',   # voiceless postalveolar fricative
    'ʒ': 'zh',   # voiced postalveolar fricative
    'θ': 'th',   # voiceless dental fricative
    'ð': 'dh',   # voiced dental fricative
    'h': 'h',    # voiceless glottal fricative
}

# Diphthong mappings (processed before individual characters)
DIPHTHONGS = {
    'eɪ': 'EY1',  # mid front tense + high front lax -> EY
    'oʊ': 'OW1',  # mid back tense + high back lax -> OW
}


def ipa_to_english(ipa_string):
    """
    Convert IPA phonetic string to English letters.

    Args:
        ipa_string (str): String in IPA notation

    Returns:
        str: English letter representation
    """
    english_string = ""
    for char in ipa_string:
        if char in IPA_TO_ENGLISH:
            english_string += IPA_TO_ENGLISH[char]
        else:
            # If character not in mapping, keep as is
            english_string += char
    return english_string


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
    data_dir = '/mnt/data/Projects/subinphon/dataset'
    csv_file = os.path.join(data_dir, 'EnglishBH_shortened_l2r_harmony.csv')
    wordlist_file = os.path.join(data_dir, 'EnglishBH_wordlist.txt')
    textgrid_file = os.path.join(data_dir, 'EnglishBH_textgrid.txt')

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return

    print(f"Reading from: {csv_file}")
    print(f"Writing wordlist to: {wordlist_file}")
    print(f"Writing textgrid to: {textgrid_file}")

    with open(csv_file, 'r', encoding='utf-8') as infile, \
         open(wordlist_file, 'w', encoding='utf-8') as wordlist_out, \
         open(textgrid_file, 'w', encoding='utf-8') as textgrid_out:

        reader = csv.DictReader(infile)

        for row in reader:
            ur_string = row['ur_string']
            ur_var = row['ur_var']

            # Wordlist output: ur_string converted to English letters, ur_var converted to ARPABET
            wordlist_col1 = ipa_to_english(ur_string)
            wordlist_col2 = ipa_to_arpabet(ur_var)
            wordlist_out.write(f"{wordlist_col1}\t{wordlist_col2}\n")

            # Textgrid output: original ur_var, English-letter version of ur_string
            textgrid_col1 = ur_var
            textgrid_col2 = ipa_to_english(ur_string)
            textgrid_out.write(f"{textgrid_col1}\t{textgrid_col2}\n")

    total_lines = sum(1 for _ in open(csv_file, 'r', encoding='utf-8')) - 1
    print(f"Processing complete. Output written to {wordlist_file} and {textgrid_file}")
    print(f"Total lines processed: {total_lines}")  # -1 for header


if __name__ == "__main__":
    main()