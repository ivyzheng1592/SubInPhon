import csv
import os


# IPA to ARPABET mapping based on common phonetic correspondences.
IPA_TO_ARPABET = {
    "i": "IY1",
    "u": "UW1",
    "ɪ": "IH1",
    "ɛ": "EH1",
    "ʊ": "UH1",
    "ɔ": "AO1",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "p": "P",
    "t": "T",
    "k": "K",
    "b": "B",
    "d": "D",
    "g": "G",
    "f": "F",
    "s": "S",
    "v": "V",
    "z": "Z",
    "ʃ": "SH",
    "ʒ": "ZH",
    "θ": "TH",
    "ð": "DH",
    "h": "HH",
}


TXT_IPA_TO_ARPABET = {
    **IPA_TO_ARPABET,
    "e": "EY1",
    "o": "OW1",
}


# IPA to English-letter mapping used by TextGrid labels.
IPA_TO_ENGLISH = {
    "i": "ii",
    "u": "uu",
    "e": "ee",
    "o": "oo",
    "ɪ": "i",
    "ɛ": "e",
    "ʊ": "u",
    "ɔ": "o",
    "m": "m",
    "n": "n",
    "ŋ": "ng",
    "p": "p",
    "t": "t",
    "k": "k",
    "b": "b",
    "d": "d",
    "g": "g",
    "f": "f",
    "s": "s",
    "v": "v",
    "z": "z",
    "ʃ": "sh",
    "ʒ": "zh",
    "θ": "th",
    "ð": "dh",
    "h": "h",
}


# Diphthong mappings are applied before per-character ARPABET conversion.
DIPHTHONGS = {
    "eɪ": "EY1",
    "oʊ": "OW1",
}


# Convert an IPA string into the English-letter form used by TextGrid labels.
def ipa_to_english(ipa_string: str) -> str:
    english_string = ""
    for char in ipa_string:
        if char in IPA_TO_ENGLISH:
            english_string += IPA_TO_ENGLISH[char]
        else:
            english_string += char
    return english_string


def _ipa_to_arpabet(
    ipa_string: str,
    ipa_to_arpabet: dict,
    use_diphthongs: bool,
) -> str:
    working_string = ipa_string
    if use_diphthongs:
        for diphthong, arpabet in DIPHTHONGS.items():
            working_string = working_string.replace(diphthong, f"[{arpabet}]")

    arpabet_phones = []
    i = 0
    while i < len(working_string):
        if working_string[i] == "[":
            end_idx = working_string.find("]", i)
            if end_idx != -1:
                arpabet_phones.append(working_string[i + 1:end_idx])
                i = end_idx + 1
                continue

        char = working_string[i]
        if char in ipa_to_arpabet:
            arpabet_phones.append(ipa_to_arpabet[char])
        else:
            arpabet_phones.append(char)
        i += 1

    return " ".join(arpabet_phones)


# Convert an audio-variant IPA string into an ARPABET phone sequence.
def aud_ipa_to_arpabet(ipa_string: str) -> str:
    return _ipa_to_arpabet(ipa_string, IPA_TO_ARPABET, use_diphthongs=True)


# Convert a text IPA string into an ARPABET phone sequence without diphthong handling.
def txt_ipa_to_arpabet(ipa_string: str) -> str:
    return _ipa_to_arpabet(ipa_string, TXT_IPA_TO_ARPABET, use_diphthongs=False)


if __name__ == "__main__":
    data_dir = "data"
    csv_file = os.path.join(data_dir, "EnglishBH_shortened_l2r_harmony.csv")
    wordlist_file = os.path.join(data_dir, "EnglishBH_wordlist.txt")
    textgrid_file = os.path.join(data_dir, "EnglishBH_textgrid.txt")

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
    else:
        print(f"Reading from: {csv_file}")
        print(f"Writing wordlist to: {wordlist_file}")
        print(f"Writing textgrid to: {textgrid_file}")

        with open(csv_file, "r", encoding="utf-8") as infile, \
             open(wordlist_file, "w", encoding="utf-8") as wordlist_out, \
             open(textgrid_file, "w", encoding="utf-8") as textgrid_out:

            reader = csv.DictReader(infile)
            for row in reader:
                ur_string = row["ur_string"]
                ur_var = row["ur_var"]

                wordlist_col1 = ipa_to_english(ur_string)
                wordlist_col2 = aud_ipa_to_arpabet(ur_var)
                wordlist_out.write(f"{wordlist_col1}\t{wordlist_col2}\n")

                textgrid_col1 = ur_var
                textgrid_col2 = ipa_to_english(ur_string)
                textgrid_out.write(f"{textgrid_col1}\t{textgrid_col2}\n")

        total_lines = sum(1 for _ in open(csv_file, "r", encoding="utf-8")) - 1
        print(f"Processing complete. Output written to {wordlist_file} and {textgrid_file}")
        print(f"Total lines processed: {total_lines}")
