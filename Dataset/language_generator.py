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

    def generate_stimuli(self, onset=None, coda=None, vowel=None, syll_struct=None, property=""):
        pass

    def decompose_stimuli(self, word_list):
        pass


class BacknessHarmony(LanguagePattern):
    def __init__(self, onset, coda, vowel, syll_struct, lang_name):
        super().__init__(onset, coda, vowel, syll_struct, lang_name)
        self.focus = self.vowel

    # function to generate vowel harmony stimuli with specified phoneme inventory and syllable structure
    def generate_stimuli(self, onset=None, coda=None, vowel=None, syll_struct=None, property=""):
        # check if there is override of phoneme inventory
        if onset is None:
            onset = self.onset
        if coda is None:
            coda = self.coda
        if vowel is None:
            vowel = self.vowel
        if syll_struct is None:
            syll_struct = self.syll_struct

        print(" - Generating stimuli:")
        # identify possible stem-suffix vowel combinations for each condition
        h_v_combinations = []
        dh_v_combinations = []
        for stem_v in vowel:
            for ur_suffix_v in vowel:
                # disallowing identical vowels, i.e. vowels w/ identical height and tenseness
                # only when generating nonidentical datasets
                if (property != "nonidentical" or
                        (property == "nonidentical" and
                        not (vowel[ur_suffix_v][1] == vowel[stem_v][1]
                        and vowel[ur_suffix_v][2] == vowel[stem_v][2]))):
                    for sr_suffix_v in vowel:
                        # harmomny
                        if (vowel[sr_suffix_v][1] == vowel[ur_suffix_v][1]
                            and vowel[sr_suffix_v][2] == vowel[ur_suffix_v][2]
                            and vowel[sr_suffix_v][0] == vowel[stem_v][0]):
                            h_v_combinations.append([stem_v, ur_suffix_v, sr_suffix_v])
                        # disharmony
                        if (vowel[sr_suffix_v][1] == vowel[ur_suffix_v][1]
                            and vowel[sr_suffix_v][2] == vowel[ur_suffix_v][2]
                            and vowel[sr_suffix_v][0] != vowel[stem_v][0]):
                            dh_v_combinations.append([stem_v, ur_suffix_v, sr_suffix_v])

        # create a dictionary with all kinds of syllables for each vowel
        syll_dict = {
            v: {key: [] for key in ["V", "CV", "VC", "CVC"]}
            for v in vowel
        }
        for v, v_dict in syll_dict.items():
            if vowel[v][2] == "tense":  # for tense vowels, only V and CV
                v_dict["V"] = [v]
                v_dict["CV"] = [o + v for o in onset]
            elif vowel[v][2] == "lax":  # for lax vowels, only VC and CVC
                v_dict["VC"] = [v + c for c in coda]
                v_dict["CVC"] = [o + v + c for o, c in itertools.product(onset, coda)]
            else:
                raise RuntimeError(f"Problem with vowel feature quality {vowel[v][2]}")

        # generate all possible vowel combinations for each syllable structure
        harmony_list = []
        disharmony_list = []
        for struct in syll_struct:
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
        harmony_file = self.lang_name + "_" + property + "_harmony.csv"
        disharmony_file = self.lang_name + "_" + property + "_disharmony.csv"

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


class FinalDevoicing(LanguagePattern):
    def __init__(self, onset, coda, vowel, syll_struct, lang_name):
        super().__init__(onset, coda, vowel, syll_struct, lang_name)
        self.focus = self.coda

    # function to generate final devoicing stimuli with specified phoneme inventory and syllable structure
    def generate_stimuli(self, onset=None, coda=None, vowel=None, syll_struct=None, property=""):
        print(" - Generating stimuli:")
        # separate voiceless and voiced codas
        coda_voiceless = [c for c in self.coda if self.coda[c] == "voiceless"]
        coda_voiced = [c for c in self.coda if self.coda[c] == "voiced"]

        devoice_list = []
        voice_list = []
        previous_devoice = []
        previous_voice = []

        # generate all possible syllables for current syllable structure
        # based on the list of syllables from previous syllable structure
        # VC -> CVC -> VCVC -> CVCVC
        for struct in self.syll_struct:
            if len(struct) == 2:  # VC
                devoice_list.extend([v + c1, v + c2] for v, (c1, c2) in
                                    itertools.product(self.vowel, zip(coda_voiceless, coda_voiceless)))
                devoice_list.extend([v + c1, v + c2] for v, (c1, c2) in
                                    itertools.product(self.vowel, zip(coda_voiced, coda_voiceless)))
                voice_list.extend([v + c1, v + c2] for v, (c1, c2) in
                                  itertools.product(self.vowel, zip(coda_voiceless, coda_voiced)))
                voice_list.extend([v + c1, v + c2] for v, (c1, c2) in
                                  itertools.product(self.vowel, zip(coda_voiced, coda_voiced)))

                previous_devoice.extend([v + c1, v + c2] for v, (c1, c2) in
                                        itertools.product(self.vowel, zip(coda_voiceless, coda_voiceless)))
                previous_devoice.extend([v + c1, v + c2] for v, (c1, c2) in
                                        itertools.product(self.vowel, zip(coda_voiced, coda_voiceless)))
                previous_voice.extend([v + c1, v + c2] for v, (c1, c2) in
                                      itertools.product(self.vowel, zip(coda_voiceless, coda_voiced)))
                previous_voice.extend([v + c1, v + c2] for v, (c1, c2) in
                                      itertools.product(self.vowel, zip(coda_voiced, coda_voiced)))
            elif struct[0] == 'V':  # VCVC
                current_devoice = [[v + ur, v + sr] for v, [ur, sr] in
                                   itertools.product(self.vowel, previous_devoice)]
                current_voice = [[v + ur, v + sr] for v, [ur, sr] in
                                 itertools.product(self.vowel, previous_voice)]

                devoice_list.extend(current_devoice)
                voice_list.extend(current_voice)

                previous_devoice = current_devoice
                previous_voice = current_voice
            else:  # CVC, CVCVC
                current_devoice = [[c + ur, c + sr] for c, [ur, sr] in
                                   itertools.product(self.onset, previous_devoice)]
                current_voice = [[c + ur, c + sr] for c, [ur, sr] in
                                 itertools.product(self.onset, previous_voice)]

                devoice_list.extend(current_devoice)
                voice_list.extend(current_voice)

                previous_devoice = current_devoice
                previous_voice = current_voice

            print(f"Now generating syllable structure {struct}, accumulating to {len(devoice_list)} pairs")
            print(f"Example {struct} devoicing pair: {devoice_list[len(devoice_list) - 1]}")
            print(f"Example {struct} voicing pair: {voice_list[len(voice_list) - 1]}")

        print(" - Writing to file:")
        devoice_file = self.lang_name + "_devoicing.csv"
        voice_file = self.lang_name + "_voicing.csv"

        with open(devoice_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header
            header = ['ur', 'sr']
            writer.writerow(header)
            # write stimuli list
            writer.writerows(devoice_list)
        print("Devoicing file ready.")

        with open(voice_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header
            header = ['ur', 'sr']
            writer.writerow(header)
            # write stimuli list
            writer.writerows(voice_list)
        print("Voicing file ready.")

        return devoice_list, voice_list

    # function to decompose final devoicing stimuli with specified phoneme inventory
    def decompose_stimuli(self, word_list):
        pass