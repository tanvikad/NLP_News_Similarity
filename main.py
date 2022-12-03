import numpy as np

import spacy

import en_core_web_sm
nlp = en_core_web_sm.load()

print("Pipeline:", nlp.pipe_names)
doc = nlp("I was reading the paper.")
token = doc[0]  # 'I'
print(token.morph)  # 'Case=Nom|Number=Sing|Person=1|PronType=Prs'
print(token.morph.get("PronType"))  # ['Prs']


from deep_translator import GoogleTranslator
translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome")  # output -> Weiter so, du bist großartig
print(translated)