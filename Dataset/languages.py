# 2025/06/05
# instances of languages together with their phoneme inventory and syllable structure


from Dataset.language_generator import *

"""
Language: English Backness Harmony
"""
# Phoneme inventory:
onset = {
    onset: None for onset in ['m', 'n', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h']
}
coda = {
    coda: None for coda in ['m', 'n', 'ŋ', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð']
}
# vowel for text input (control for number of symbols in a phoneme)
vowel_txt = {'i': ["front", "high", "tense"],
                'e': ["front", "mid", "tense"],
                'u': ["back", "high", "tense"],
                'o': ["back", "mid", "tense"],
                'ɪ': ["front", "high", "lax"],
                'ɛ': ["front", "mid", "lax"],
                'ʊ': ["back", "high", "lax"],
                'ɔ': ["back", "mid", "lax"]}
# vowel for audio input (actual realization of phoneme)
vowel_aud = {'i': ["front", "high", "tense"],
                'eɪ': ["front", "mid", "tense"],
                'u': ["back", "high", "tense"],
                'oʊ': ["back", "mid", "tense"],
                'ɪ': ["front", "high", "lax"],
                'ɛ': ["front", "mid", "lax"],
                'ʊ': ["back", "high", "lax"],
                'ɔ': ["back", "mid", "lax"]}
# Syllable structure:
syll_struct = ["V-CV", "V-CVC", "CV-CV", "CV-CVC",
               "VC-V", "VC-VC", "CVC-V", "CVC-VC"]
syll_struct_shortened = ["V-CV", "V-CVC", "CV-CV",
                         "VC-V", "VC-VC", "CVC-V"]
# Stimuli
EnglishBH_full_txt = BacknessHarmony(onset, coda, vowel_txt, syll_struct,
                                "EnglishBH_full_txt")
#EnglishBH_full_txt.generate_stimuli()
EnglishBH_nonidentical_txt = BacknessHarmony(onset, coda, vowel_txt, syll_struct,
                                             "EnglishBH_nonidentical_txt")
#EnglishBH_nonidentical_txt.generate_stimuli()
EnglishBH_full_fea = BacknessHarmony(onset, coda, vowel_txt, syll_struct,
                                "EnglishBH_full_fea")
#EnglishBH_full_fea.generate_stimuli()
EnglishBH_nonidentical_fea = BacknessHarmony(onset, coda, vowel_txt, syll_struct,
                                             "EnglishBH_nonidentical_fea")
#EnglishBH_nonidentical_fea.generate_stimuli()
EnglishBH_shortened_txt = BacknessHarmony(onset, coda, vowel_txt, syll_struct_shortened,
                                          "EnglishBH_shortened_txt")
#EnglishBH_shortened_txt.generate_stimuli()
EnglishBH_shortened_aud = BacknessHarmony(onset, coda, vowel_txt, syll_struct_shortened,
                                          "EnglishBH_shortened_aud")
#EnglishBH_shortened_aud.generate_stimuli()


"""
Language: English Final Devoicing
"""
# Phoneme inventory:
onset = {
    onset: None for onset in ['p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð']
}
coda = {
    voiceless: "voiceless" for voiceless in ['p', 't', 'k', 'f', 's', 'ʃ', 'θ']
}
coda.update({
    voiced: "voiced" for voiced in ['b', 'd', 'g', 'v', 'z', 'ʒ', 'ð']
})
# vowel for text input (control for number of symbols in a phoneme)
vowel_txt = {
    vowel: None for vowel in ['i', 'e', 'u', 'o', 'ɪ', 'ɛ', 'ʊ', 'ɔ']
}
# vowel for audio input (actual realization of phoneme)
vowel_aud = {
    vowel: None for vowel in ['i', 'eɪ', 'u', 'oʊ', 'ɪ', 'ɛ', 'ʊ', 'ɔ']
}
# Syllable structure:
syll_struct = ["VC", "CVC", "VCVC", "CVCVC"]
# Stimuli
#EnglishFD_txt = FinalDevoicing(onset, coda, vowel_txt, syll_struct,
                               #"EnglishFD_txt")
#EnglishFD_txt.generate_stimuli()
#EnglishFD_fea = FinalDevoicing(onset, coda, vowel_txt, syll_struct,
                               #"EnglishFD_fea")
#EnglishFD_fea.generate_stimuli()
#EnglishFD_aud = FinalDevoicing(onset, coda, vowel_aud, syll_struct,
                               #"EnglishFD_aud")
#EnglishFD_aud.generate_stimuli()


# Language: Cantonese Backness Harmony
# Phoneme inventory:
onset = {
    onset: None for onset in ['m', 'n', 'ng', 'p', 't', 'k', 'b', 'd', 'g', 'z', 'c', 's', 'f', 'h']
}
coda = {
    coda: None for coda in ['m', 'n', 'ng', 'p', 't', 'k']
}
vowel = {'i': ["front", "high", "tense"],
         'e': ["front", "mid", "tense"],
         'u': ["back", "high", "tense"],
         'o': ["back", "mid", "tense"]}
# Syllable structure:
syll_struct = ["V-CV", "V-CVC", "CV-CV", "CV-CVC",
               "V.CV-CV", "V.CV-CVC", "CV.CV-CV", "CV.CV-CVC"]
# Stimuli
#CantoneseBH = BacknessHarmony(onset_c, coda_c, vowel_c, syll_struct_c, "CantoneseBH")
#CantoneseBH.generate_stimuli()


languages = {
    "EnglishBH_full_txt": EnglishBH_full_txt,
    "EnglishBH_nonidentical_txt": EnglishBH_nonidentical_txt,
    "EnglishBH_full_fea": EnglishBH_full_fea,
    "EnglishBH_nonidentical_fea": EnglishBH_nonidentical_fea,
    "EnglishBH_shortened_aud": EnglishBH_shortened_aud,
    #"EnglishFD_txt": EnglishFD_txt,
    #"EnglishFD_fea": EnglishFD_fea,
    #"EnglishFD_aud": EnglishFD_aud
}
