# created 2024/09/15
# A script to generate stimuli of vowel harmony and vowel disharmony
# based on given phoneme inventory and syllable structures
# updated 2025/04/04
# added function to decompose vowel harmony and vowel disharmony stimuli
# into structured syllables based on given phoneme inventory and syllable structures

import csv
import itertools
from os import remove


# Function to generate stimuli for each language with specified phoneme inventory and syllable structure
def generate_stimuli(onset, coda, vowel_1, vowel_2, syll_struct, language):
    print(" - Generating stimuli:")
    harmony_list = []
    disharmony_list = []

    # separate vowels that can occur in open/closed syllables
    vowel_open_1 = [key for key, value in vowel_1.items() if value == "open"]
    vowel_close_1 = [key for key, value in vowel_1.items() if value == "closed"]
    if not vowel_close_1:
        vowel_close_1 = vowel_open_1
    vowel_open_2 = [key for key, value in vowel_2.items() if value == "open"]
    vowel_close_2 = [key for key, value in vowel_2.items() if value == "closed"]
    if not vowel_close_2:
        vowel_close_2 = vowel_open_2

    # create two dictionaries with all kinds of syllables
    syll_dict_1 = {key: None for key in ["V", "VC", "CV", "CVC"]}
    syll_dict_1["V"] = [v for v in vowel_open_1]
    syll_dict_1["VC"] = [v + c for v, c in itertools.product(vowel_close_1, coda)]
    syll_dict_1["CV"] = [o + v for o, v in itertools.product(onset, vowel_open_1)]
    syll_dict_1["CVC"] = [o + v + c for o, v, c in itertools.product(onset, vowel_close_1, coda)]
    syll_dict_2 = {key: None for key in ["V", "VC", "CV", "CVC"]}
    syll_dict_2["V"] = [v for v in vowel_open_2]
    syll_dict_2["VC"] = [v + c for v, c in itertools.product(vowel_close_2, coda)]
    syll_dict_2["CV"] = [o + v for o, v in itertools.product(onset, vowel_open_2)]
    syll_dict_2["CVC"] = [o + v + c for o, v, c in itertools.product(onset, vowel_close_2, coda)]

    # for each syllable structure
    for struct in syll_struct:
        # separate stem and suffix in each syllable structure
        stem_struct = struct.split("-")[0]
        suffix_struct = struct.split("-")[1]

        # get all possible stems
        if "." in stem_struct:  # if stem is disyllabic
            # stem_1 has vowel_1 in second syllable
            stem_1 = [s_1 + s_2 for s_1, s_2 in
                      itertools.product(syll_dict_1[stem_struct.split(".")[0]] + syll_dict_2[stem_struct.split(".")[0]],
                                        syll_dict_1[stem_struct.split(".")[1]])]
            # stem_2 has vowel_2 in second syllable
            stem_2 = [s_1 + s_2 for s_1, s_2 in
                      itertools.product(syll_dict_1[stem_struct.split(".")[0]] + syll_dict_2[stem_struct.split(".")[0]],
                                        syll_dict_2[stem_struct.split(".")[1]])]
        else:  # if stem is monosyllabic
            stem_1 = syll_dict_1[stem_struct]
            stem_2 = syll_dict_2[stem_struct]

        # get all possible suffixes
        suffix_1 = syll_dict_1[suffix_struct]
        suffix_2 = syll_dict_2[suffix_struct]

        # combine stem and suffixes
        harmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                            itertools.product(stem_1, zip(suffix_1, suffix_1)))
        harmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                            itertools.product(stem_1, zip(suffix_2, suffix_1)))
        harmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                            itertools.product(stem_2, zip(suffix_1, suffix_2)))
        harmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                            itertools.product(stem_2, zip(suffix_2, suffix_2)))
        disharmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                               itertools.product(stem_1, zip(suffix_1, suffix_2)))
        disharmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                               itertools.product(stem_1, zip(suffix_2, suffix_2)))
        disharmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                               itertools.product(stem_2, zip(suffix_1, suffix_1)))
        disharmony_list.extend([stem, stem + suffix_ur, stem + suffix_sr] for stem, (suffix_ur, suffix_sr) in
                               itertools.product(stem_2, zip(suffix_2, suffix_1)))
        print(f"Now generating syllable structure {struct}, accumulating to {len(harmony_list)} pairs")
        print(f"Example {struct} harmony pair: {harmony_list[len(harmony_list)-1]}")
        print(f"Example {struct} disharmony pair: {disharmony_list[len(disharmony_list) - 1]}")

    print(" - Writing to file:")
    harmony_file = language + "_harmony.csv"
    disharmony_file = language + "_disharmony.csv"

    with open(harmony_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write header
        header = ['stem', 'ur', 'sr']
        writer.writerow(header)
        # write stimuli list
        writer.writerows(harmony_list)
    print("Harmony file ready.")

    with open(disharmony_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write header
        header = ['stem', 'ur', 'sr']
        writer.writerow(header)
        # write stimuli list
        writer.writerows(disharmony_list)
    print("Disharmony file ready.")

    return harmony_list, disharmony_list


# Function to decompose stimuli for each language with specified phoneme inventory
def decompose_stimuli(onset, coda, vowel_1, vowel_2, word):

    syll1 = [None, None, None]  # C, V, C
    syll2 = [None, None, None]

    vowel_open_1 = [key for key, value in vowel_1.items() if value == "open"]
    vowel_close_1 = [key for key, value in vowel_1.items() if value == "closed"]
    vowel_open_2 = [key for key, value in vowel_2.items() if value == "open"]
    vowel_close_2 = [key for key, value in vowel_2.items() if value == "closed"]

    if word and word[0] == "<SOS>":
    # remove start of sentence token
        word.pop(0)
    if word and word[0] in onset:
    # if first syllable has onset
        syll1[0] = word[0]
        word.pop(0)
    while word and word[0] not in vowel_1 and word[0] not in vowel_2:
    # if first phone is coda or next phone(s) are onset/coda
    # get rid of the phone until we meet a vowel
        syll1[0] = False
        word.pop(0)

    if word and (word[0] in vowel_open_1 or word[0] in vowel_open_2):
    # if first vowel is in an open syllable
        syll1[1] = word[0]
        word.pop(0)
        while word and word[0] not in onset:
        # the syllable needs to be immediately followed by an onset
            syll1[2] = False
            word.pop(0)
    elif word and (word[0] in vowel_close_1 or word[0] in vowel_close_2):
    # if first vowel is in a closed syllable
        syll1[1] = word[0]
        word.pop(0)
        if word and word[0] in coda:
        # the syllable needs a coda
            syll1[2] = word[0]
            word.pop(0)
        while word and word[0] not in vowel_1 and word[0] not in vowel_2:
            # the next syllable cannot have an onset
            # get rid of the phone until we meet a vowel
                syll2[0] = False
                word.pop(0)
        else:
            syll1[2] = False
    else:
    # if first vowel looks weird
        syll1[1] = False
        while word and word[0] not in onset:
            # get rid of the phone until we meet the next syllable onset
            word.pop(0)

    if word and word[0] in onset:
    # if second syllable has onset
        syll2[0] = word[0]
        word.pop(0)
    while word and word[0] not in vowel_1 and word[0] not in vowel_2:
    # if next phone(s) are onset/coda
    # get rid of the phone until we meet a vowel
        syll2[0] = False
        word.pop(0)

    if word and (word[0] in vowel_open_1 or word[0] in vowel_open_2):
    # if second vowel is in an open syllable
        syll2[1] = word[0]
        word.pop(0)
    elif word and (word[0] in vowel_close_1 or word[0] in vowel_close_2):
    # if second vowel is in a closed syllable
        syll2[1] = word[0]
        word.pop(0)
        if word and word[0] in coda:
        # the syllable needs a coda
            syll2[2] = word[0]
            word.pop(0)
        else:
            syll2[2] = False
    else:
    # if the second vowel looks weird
        syll2[1] = False

    if word and word[0] != "<EOS>":  # there should be no more phones
        syll2[2] = False

    sylls = [syll1, syll2]
    return sylls


# Language: English
# Phoneme inventory:
onset_ae = ['m', 'n', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h']
coda_ae = ['m', 'n', 'ŋ', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð']
# vowel for text input (control for number of symbols in a phoneme) -> abandoned
vowel_front_ae_txt = {'ɪ': "closed", 'ɛ': "closed", 'i': "open", 'e': "open"}
vowel_back_ae_txt = {'ʊ': "closed", 'ɔ': "closed", 'u': "open", 'o': "open"}
# vowel for audio input (actual realization of phoneme)
vowel_front_ae_aud = {'ɪ': "closed", 'ɛ': "closed", 'i': "open", 'eɪ': "open"}
vowel_back_ae_aud = {'ʊ': "closed", 'ɔ': "closed", 'u': "open", 'oʊ': "open"}
# Syllable structure:
syll_struct_ae = ["V-CV", "V-CVC", "CV-CV", "CV-CVC",
                  "VC-V", "VC-VC", "CVC-V", "CVC-VC"]
                  #"V.CV-CV", "V.CV-CVC", "V.CVC-V", "V.CVC-VC",
                  #"CV.CV-CV", "CV.CV-CVC", "CV.CVC-V", "CV.CVC-VC",
                  #"VC.V-CV", "VC.V-CVC", "VC.VC-V", "VC.VC-VC",
                  #"CVC.V-CV", "CVC.V-CVC","CVC.VC-V", "CVC.VC-VC"]
# Stimuli
#generate_stimuli(onset_ae, coda_ae, vowel_front_ae_txt, vowel_back_ae_txt, syll_struct_ae, "EnglishBH_txt")
#generate_stimuli(onset_ae, coda_ae, vowel_front_ae_aud, vowel_back_ae_aud, syll_struct_ae, "EnglishBH_aud")

# Language: Cantonese
# Phoneme inventory:
onset_c = ['m', 'n', 'ng', 'p', 't', 'k', 'b', 'd', 'g', 'z', 'c', 's', 'f', 'h']
coda_c = ['m', 'n', 'ng', 'p', 't', 'k']
vowel_front_c = {'i': "open", 'e': "open"}
vowel_back_c = {'o': "open", 'u': "open"}
# Syllable structure:
syll_struct_c = ["V-CV", "V-CVC", "CV-CV", "CV-CVC",
                 #"VC-V", "VC-VC", "CVC-V", "CVC-VC",
                 "V.CV-CV", "V.CV-CVC", "CV.CV-CV", "CV.CV-CVC"]
                 #"VC.V-CV", "VC.V-CVC", "CVC.V-CV", "CVC.V-CVC"
                 #"V.CVC-V", "CV.CVC-VC", "CV.CVC-V", "CV.CVC-VC"
                 #"VC.VC-V", "VC.VC-VC", "CVC.VC-V", "CVC.VC-VC"]
# Stimuli
#generate_stimuli(onset_c, coda_c, vowel_front_c, vowel_back_c, syll_struct_c, "Cantonese")
