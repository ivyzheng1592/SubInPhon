# created 2024/09/15
# updated 2025/06/05
# class of language patterns
# with information of the phoneme inventory and syllable structure of a language
# together with functions to generate and decompose stimuli for this language

import csv
import itertools


class LanguagePattern:
    def __init__(self, onset, coda, vowel, syll_struct, lang_name):

        self.lang_name = lang_name
        self.syll_struct = syll_struct
        self.onset = onset
        self.coda = coda
        self.vowel = vowel
        self.focus = {}  # different focus for different language pattern


class BacknessHarmony(LanguagePattern):
    def __init__(self, onset, coda, vowel, syll_struct, lang_name):
        super().__init__(onset, coda, vowel, syll_struct, lang_name)
        self.focus = self.vowel

    # function to generate vowel harmony stimuli with specified phoneme inventory and syllable structure
    def generate_stimuli(self):
        print(" - Generating stimuli:")

        # identify possible stem-suffix vowel combinations for each condition
        h_v_combinations = []
        dh_v_combinations = []
        for stem_v in self.vowel:
            for ur_suffix_v in self.vowel:
                # disallowing identical vowels, i.e. vowels w/ identical height and tenseness
                if not (self.vowel[ur_suffix_v][1] == self.vowel[stem_v][1]
                        and self.vowel[ur_suffix_v][2] == self.vowel[stem_v][2]):
                    for sr_suffix_v in self.vowel:
                        # harmomny
                        if (self.vowel[sr_suffix_v][1] == self.vowel[ur_suffix_v][1]
                            and self.vowel[sr_suffix_v][2] == self.vowel[ur_suffix_v][2]
                            and self.vowel[sr_suffix_v][0] == self.vowel[stem_v][0]):
                            h_v_combinations.append([stem_v, ur_suffix_v, sr_suffix_v])
                        # disharmony
                        if (self.vowel[sr_suffix_v][1] == self.vowel[ur_suffix_v][1]
                            and self.vowel[sr_suffix_v][2] == self.vowel[ur_suffix_v][2]
                            and self.vowel[sr_suffix_v][0] != self.vowel[stem_v][0]):
                            dh_v_combinations.append([stem_v, ur_suffix_v, sr_suffix_v])

        # create a dictionary with all kinds of syllables for each vowel
        syll_dict = {
            v: {key: [] for key in ["V", "VC", "CV", "CVC"]}
            for v in self.vowel
        }
        for v, v_dict in syll_dict.items():
            if self.vowel[v][2] == "tense":  # for tense vowels, only V and CV
                v_dict["V"] = [v]
                v_dict["CV"] = [o + v for o in self.onset]
            elif self.vowel[v][2] == "lax":  # for lax vowels, only VC and CVC
                v_dict["VC"] = [v + c for c in self.coda]
                v_dict["CVC"] = [o + v + c for o, c in itertools.product(self.onset, self.coda)]
            else:
                raise RuntimeError(f"Incorrect vowel feature value {self.vowel[v][2]}")

        # generate all possible vowel combinations for each syllable structure
        harmony_list = []
        disharmony_list = []
        for struct in self.syll_struct:
            # separate stem and suffix
            # assume monosyllabic stem
            stem_struct = struct.split("-")[0]
            suffix_struct = struct.split("-")[1]

            for [stem_v, ur_suffix_v, sr_suffix_v] in h_v_combinations:
                # if the syllable structure exist for the current vowel
                if (syll_dict[stem_v][stem_struct]
                    and syll_dict[ur_suffix_v][suffix_struct]):

                    # combine stem and suffixes
                    stems = syll_dict[stem_v][stem_struct]
                    ur_suffixes = syll_dict[ur_suffix_v][suffix_struct]
                    sr_suffixes = syll_dict[sr_suffix_v][suffix_struct]

                    harmony_list.extend([stem, stem + ur_suffix, stem + sr_suffix]
                                        for stem, (ur_suffix, sr_suffix) in
                                        itertools.product(stems, zip(ur_suffixes, sr_suffixes)))

            for [stem_v, ur_suffix_v, sr_suffix_v] in dh_v_combinations:
                # if the syllable structure exist for the current vowel
                if (syll_dict[stem_v][stem_struct]
                        and syll_dict[ur_suffix_v][suffix_struct]):
                    # combine stem and suffixes
                    stems = syll_dict[stem_v][stem_struct]
                    ur_suffixes = syll_dict[ur_suffix_v][suffix_struct]
                    sr_suffixes = syll_dict[sr_suffix_v][suffix_struct]

                    disharmony_list.extend([stem, stem + ur_suffix, stem + sr_suffix]
                                           for stem, (ur_suffix, sr_suffix) in
                                           itertools.product(stems, zip(ur_suffixes, sr_suffixes)))

            print(f"Now generating syllable structure {struct}, accumulating to {len(harmony_list)} pairs")
            print(f"Example {struct} harmony pair: {harmony_list[len(harmony_list) - 1]}")
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

        # separate tense and lax vowels
        vowel_tense = [v for v in self.vowel if self.vowel[v][2] == "tense"]
        vowel_lax = [v for v in self.vowel if self.vowel[v][2] == "lax"]

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

        # get possible syllable structures combining stem and suffix
        syll_struct = [struct.replace("-", "") for struct in self.syll_struct]
        syll_struct = list(set(syll_struct))  # remove duplicates

        # check syllable structure of word
        word_struct = ""
        for char in word_copy:
            if char in self.onset or char in self.coda:
                word_struct = word_struct + "C"
            elif char in self.vowel:
                word_struct = word_struct + "V"
            else:
                word_struct = "False"
        if word_struct not in syll_struct:
            syll1[0] = False
            syll2[0] = False
            return sylls

        # if first syllable has onset
        if word_copy and (word_copy[0] in self.coda or word_copy[0] in self.onset):
            syll1[0] = word_copy[0]
            word_copy.pop(0)

        # if first syllable has close vowel
        if word_copy and word_copy[0] in vowel_lax:
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
                raise RuntimeError(f"Problem with output recording {word_list}")
        # if first syllable has open vowel
        elif word_copy and word_copy[0] in vowel_tense:
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
                raise RuntimeError(f"Problem with output recording {word_list}")
        else:
            raise RuntimeError(f"Problem with output recording {word_list}")

        # the second syllable should have vowel
        if word_copy and (word_copy[0] in self.vowel):
            syll2[1] = word_copy[0]
            word_copy.pop(0)
        else:
            raise RuntimeError(f"Problem with output recording {word_list}")

        # the following consonant should be coda
        if word_copy and (word_copy[0] in self.coda or word_copy[0] in self.onset):
            syll2[2] = word_copy[0]
            word_copy.pop(0)
        elif word_copy:
            raise RuntimeError(f"Problem with output recording {word_list}")

        return sylls
