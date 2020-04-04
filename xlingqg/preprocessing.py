import spacy
import neuralcoref
from typing import List

class Prepocessor():

    def __init__(self):
        self.nlp  = nlp = spacy.load('en_core_web_lg')
        neuralcoref.add_to_pipe(self.nlp)

    def preprocess(self, text):
        text_with_resolved_corefs = self.__resolve_corefs(text)
        doc = self.nlp(text_with_resolved_corefs)
        return doc

    def __resolve_corefs(self,text):
        doc = self.nlp(text)
        return doc._.coref_resolved

