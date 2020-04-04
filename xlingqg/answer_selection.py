from typing import List

class AnswerSelectionRule():
    def extract_answer(self, sentence) -> List[List[int]]:
        pass

class NounPhraseSelector(AnswerSelectionRule):

    def extract_answer(self, sentence) -> List[List[int]]:
        answers = []
        for noun_phrase in sentence.noun_chunks:
            if noun_phrase.root.tag_  == 'PRP': continue
            answer_indices = [subtree_token.i-sentence.start for subtree_token in noun_phrase.subtree]
            answers.append(answer_indices)
        return answers      

class ComplementSelector(AnswerSelectionRule):

    def extract_answer(self, sentence) -> List[List[int]]:
        answers = []
        for token in sentence:
            if token.dep_  in ('ccomp','xcomp'):
                answer_indices = [subtree_token.i-sentence.start for subtree_token in token.subtree]
                answers.append(answer_indices)
        return answers  

class SubjectSelector(AnswerSelectionRule):

    def extract_answer(self, sentence) -> List[List[int]]:
        answers = []
        for token in sentence:
            if token.dep_  in ('agent','nsubj','nsubjpass','csubj','csubjpass'):
                if token.tag_ == 'PRP': continue
                answer_indices = [subtree_token.i-sentence.start for subtree_token in token.subtree]
                answers.append(answer_indices)
        return answers  

class ObjectSelector(AnswerSelectionRule):
    def extract_answer(self, sentence) -> List[List[int]]:
        answers = []
        for token in sentence:
            if token.dep_  in ('attr','dobj','obj'):
                if token.tag_ == 'PRP': continue
                answer_indices = [subtree_token.i-sentence.start for subtree_token in token.subtree]
                answers.append(answer_indices)
        return answers    


class AdverbialClausModifierSelector(AnswerSelectionRule):

    def extract_answer(self, sentence) -> List[List[int]]:
        answers = []
        for token in sentence:
            if token.dep_  == 'advcl':
                answer_indices = [subtree_token.i-sentence.start for subtree_token in token.subtree]
                answers.append(answer_indices)
        return answers              

class PrepositionalModifierSelector(AnswerSelectionRule):

    def extract_answer(self, sentence) -> List[List[int]]:
        answers = []
        for token in sentence:
            if token.dep_ == 'prep':
                answer_indices = [subtree_token.i-sentence.start for subtree_token in token.subtree]
                answers.append(answer_indices)
        return answers       


class AnswerSelector():
    
    def __init__(self):
        self.rules = [ObjectSelector(), SubjectSelector(), PrepositionalModifierSelector()]

    def select_answers(self,doc): 
        sentence_answers=[]
        for sentence in doc.sents:
            answers = []
            for rule in self.rules:
                answer_indices = rule.extract_answer(sentence)
                if answer_indices: answers.extend(answer_indices)
            if answers: sentence_answers.append((sentence,answers))
        return sentence_answers