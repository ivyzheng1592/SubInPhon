# created 2024/09/15
# updated 2025/06/05
# class of language patterns
# with information of the phoneme inventory and syllable structure of a language
# together with functions to generate and decompose stimuli for this language

import csv
import itertools


class LanguagePattern:
    def __init__(self, onset, coda, vowel_1, vowel_2, syll_struct, lang_name):

        self.lang_name = lang_name
        self.syll_struct_sep = syll_struct  # syllable structure separated into stem-suffix
        self.syll_struct_com = [struct.replace("-", "") for struct in self.syll_struct_sep]  # syllable structure combining stem and suffix
        self.syll_struct = list(set(self.syll_struct_com))  # remove duplicates

        self.onset = onset
        self.coda = coda
        self.consonant = onset + coda
        self.vowel_1 = vowel_1
        self.vowel_2 = vowel_2
        self.vowel = {**vowel_1, **vowel_2}
        self.phone = self.consonant + list(self.vowel.keys())


class BacknessHarmony(LanguagePattern):
    def __init__(self, onset, coda, vowel_1, vowel_2, syll_struct, lang_name):
        super().__init__(onset, coda, vowel_1, vowel_2, syll_struct, lang_name)

    # function to generate vowel harmony stimuli with specified phoneme inventory and syllable structure
    def generate_stimuli(self):
        print(" - Generating stimuli:")
        harmony_list = []
        disharmony_list = []

        # separate vowels that can occur in open/closed syllables
        vowel_open_1 = [key for key, value in self.vowel_1.items() if value == "open"]
        vowel_close_1 = [key for key, value in self.vowel_1.items() if value == "closed"]
        if not vowel_close_1:
            vowel_close_1 = vowel_open_1
        vowel_open_2 = [key for key, value in self.vowel_2.items() if value == "open"]
        vowel_close_2 = [key for key, value in self.vowel_2.items() if value == "closed"]
        if not vowel_close_2:
            vowel_close_2 = vowel_open_2

        # create two dictionaries with all kinds of syllables
        syll_dict_1 = {key: None for key in ["V", "VC", "CV", "CVC"]}
        syll_dict_1["V"] = [v for v in vowel_open_1]
        syll_dict_1["VC"] = [v + c for v, c in itertools.product(vowel_close_1, self.coda)]
        syll_dict_1["CV"] = [o + v for o, v in itertools.product(self.onset, vowel_open_1)]
        syll_dict_1["CVC"] = [o + v + c for o, v, c in itertools.product(self.onset, vowel_close_1, self.coda)]
        syll_dict_2 = {key: None for key in ["V", "VC", "CV", "CVC"]}
        syll_dict_2["V"] = [v for v in vowel_open_2]
        syll_dict_2["VC"] = [v + c for v, c in itertools.product(vowel_close_2, self.coda)]
        syll_dict_2["CV"] = [o + v for o, v in itertools.product(self.onset, vowel_open_2)]
        syll_dict_2["CVC"] = [o + v + c for o, v, c in itertools.product(self.onset, vowel_close_2, self.coda)]

        # for each syllable structure
        for struct in self.syll_struct_sep:
            # separate stem and suffix
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
        harmony_file = self.lang_name + "_harmony.csv"
        disharmony_file = self.lang_name + "_disharmony.csv"

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


    # function to decompose vowel harmony stimuli with specified phoneme inventory
    def decompose_stimuli(self, word_list):

        # separate vowels that can occur in open/closed syllables
        vowel_open = [key for key, value in self.vowel.items() if value == "open"]
        vowel_close = [key for key, value in self.vowel.items() if value == "closed"]

        word_copy = word_list.copy()  # copy of word for token removal
        syll1 = [None, None, None]  # C, V, C
        syll2 = [None, None, None]  # C, V ,C
        sylls = [syll1, syll2]

        # remove SOS token in word
        if word_copy and word_copy[0] == "<SOS>":
            word_copy.pop(0)
        # remove PAD token in word
        while word_copy and word_copy[-1] == "<PAD>":
            word_copy.pop()
        # remove EOS token in word
        while word_copy and word_copy[-1] == "<EOS>":
            word_copy.pop()

        # check syllable structure of word
        word_struct = ""
        for char in word_copy:
            if char in self.onset or char in self.coda:
                word_struct = word_struct + "C"
            elif char in self.vowel_1 or char in self.vowel_2:
                word_struct = word_struct + "V"
            else:
                word_struct = "False"
        if word_struct not in self.syll_struct_com:
            syll1[0] = False
            syll2[0] = False
            return sylls

        # if first syllable has onset
        if word_copy and (word_copy[0] in self.coda or word_copy[0] in self.onset):
            syll1[0] = word_copy[0]
            word_copy.pop(0)

        # if first syllable has close vowel
        if word_copy and word_copy[0] in vowel_close:
            syll1[1] = word_copy[0]
            word_copy.pop(0)
            # the following consonant should be coda
            if word_copy and word_copy[0] in self.coda:
                syll1[2] = word_copy[0]
                word_copy.pop(0)
            elif word_copy and word_copy[0] in self.onset:
                syll2[0] = word_copy[0]
                word_copy.pop(0)
            else:
                raise RuntimeError(f"Problem with output recording.{word_list}")
        # if first syllable has open vowel
        elif word_copy and word_copy[0] in vowel_open:
            syll1[1] = word_copy[0]
            word_copy.pop(0)
            # the following consonant should be onset
            if word_copy and word_copy[0] in self.onset:
                syll2[0] = word_copy[0]
                word_copy.pop(0)
            elif word_copy and word_copy[0] in self.coda:
                syll1[2] = word_copy[0]
                word_copy.pop(0)
            else:
                raise RuntimeError(f"Problem with output recording.{word_list}")
        else:
            raise RuntimeError(f"Problem with output recording.{word_list}")

        # the second syllable should have vowel
        if word_copy and (word_copy[0] in vowel_close or word_copy[0] in vowel_open):
            syll2[1] = word_copy[0]
            word_copy.pop(0)
        else:
            raise RuntimeError(f"Problem with output recording.{word_list}")

        # the following consonant should be coda
        if word_copy and (word_copy[0] in self.coda or word_copy[0] in self.onset):
            syll2[2] = word_copy[0]
            word_copy.pop(0)
        elif word_copy:
            raise RuntimeError(f"Problem with output recording.{word_list}")

        return sylls
