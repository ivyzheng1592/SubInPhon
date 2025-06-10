# 2025/06/05
# instances of languages together with their phoneme inventory and syllable structure


from language_generator import *


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
EnglishBH_txt = BacknessHarmony(onset_ae, coda_ae, vowel_front_ae_txt, vowel_back_ae_txt,
                                syll_struct_ae, "EnglishBH_txt")
#EnglishBH_txt.generate_stimuli()
EnglishBH_aud = BacknessHarmony(onset_ae, coda_ae, vowel_front_ae_aud, vowel_back_ae_aud,
                                syll_struct_ae, "EnglishBH_aud")
#EnglishBH_aud.generate_stimuli()


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
CantoneseBH = BacknessHarmony(onset_c, coda_c, vowel_front_c, vowel_back_c, syll_struct_c, "CantoneseBH")
#CantoneseBH.generate_stimuli()


languages = {
    "EnglishBH_txt": EnglishBH_txt,
    "EnglishBH_aud": EnglishBH_aud,
    "CantoneseBH": CantoneseBH
}
