# 2025/06/05
# instances of languages together with their phoneme inventory and syllable structure


from Dataset.language_generator import *


# Language: English
# Phoneme inventory:
onset_ae = {
    onset: None for onset in ['m', 'n', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð', 'h']
}
coda_ae = {
    coda: None for coda in ['m', 'n', 'ŋ', 'p', 't', 'k', 'b', 'd', 'g', 'f', 's', 'v', 'z', 'ʃ', 'ʒ', 'θ', 'ð']
}
# vowel for text input (control for number of symbols in a phoneme) -> abandoned
vowel_ae_txt = {'i': ["front", "high", "tense"],
                'e': ["front", "mid", "tense"],
                'u': ["back", "high", "tense"],
                'o': ["back", "mid", "tense"],
                'ɪ': ["front", "high", "lax"],
                'ɛ': ["front", "mid", "lax"],
                'ʊ': ["back", "high", "lax"],
                'ɔ': ["back", "mid", "lax"]}
# vowel for audio input (actual realization of phoneme)
vowel_ae_aud = {'i': ["front", "high", "tense"],
                'eɪ': ["front", "mid", "tense"],
                'u': ["back", "high", "tense"],
                'oʊ': ["back", "mid", "tense"],
                'ɪ': ["front", "high", "lax"],
                'ɛ': ["front", "mid", "lax"],
                'ʊ': ["back", "high", "lax"],
                'ɔ': ["back", "mid", "lax"]}
# Syllable structure:
syll_struct_ae = ["V-CV", "V-CVC", "CV-CV", "CV-CVC",
                  "VC-V", "VC-VC", "CVC-V", "CVC-VC"]
                  #"V.CV-CV", "V.CV-CVC", "V.CVC-V", "V.CVC-VC",
                  #"CV.CV-CV", "CV.CV-CVC", "CV.CVC-V", "CV.CVC-VC",
                  #"VC.V-CV", "VC.V-CVC", "VC.VC-V", "VC.VC-VC",
                  #"CVC.V-CV", "CVC.V-CVC","CVC.VC-V", "CVC.VC-VC"]
# Stimuli
EnglishBH_txt = BacknessHarmony(onset_ae, coda_ae, vowel_ae_txt, syll_struct_ae, "EnglishBH_txt")
#EnglishBH_txt.generate_stimuli()
EnglishBH_fea = BacknessHarmony(onset_ae, coda_ae, vowel_ae_txt, syll_struct_ae, "EnglishBH_fea")
#EnglishBH_fea.generate_stimuli()
EnglishBH_aud = BacknessHarmony(onset_ae, coda_ae, vowel_ae_aud, syll_struct_ae, "EnglishBH_aud")
#EnglishBH_aud.generate_stimuli()


# Language: Cantonese
# Phoneme inventory:
onset_c = {
    onset: None for onset in ['m', 'n', 'ng', 'p', 't', 'k', 'b', 'd', 'g', 'z', 'c', 's', 'f', 'h']
}
coda_c = {
    coda: None for coda in ['m', 'n', 'ng', 'p', 't', 'k']
}
vowel_c = {'i': ["front", "high", "tense"],
           'e': ["front", "mid", "tense"],
           'u': ["back", "high", "tense"],
           'o': ["back", "mid", "tense"]}
# Syllable structure:
syll_struct_c = ["V-CV", "V-CVC", "CV-CV", "CV-CVC",
                 #"VC-V", "VC-VC", "CVC-V", "CVC-VC",
                 "V.CV-CV", "V.CV-CVC", "CV.CV-CV", "CV.CV-CVC"]
                 #"VC.V-CV", "VC.V-CVC", "CVC.V-CV", "CVC.V-CVC"
                 #"V.CVC-V", "CV.CVC-VC", "CV.CVC-V", "CV.CVC-VC"
                 #"VC.VC-V", "VC.VC-VC", "CVC.VC-V", "CVC.VC-VC"]
# Stimuli
#CantoneseBH = BacknessHarmony(onset_c, coda_c, vowel_c, syll_struct_c, "CantoneseBH")
#CantoneseBH.generate_stimuli()


languages = {
    "EnglishBH_txt": EnglishBH_txt,
    "EnglishBH_fea": EnglishBH_fea,
    "EnglishBH_aud": EnglishBH_aud,
}
