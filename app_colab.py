import xlingqg
from flask import Flask, jsonify, request
import waitress

def main():
    qg = xlingqg.CrossLingualQuestionGenerator()
    preprocessor = xlingqg.Prepocessor()
    answer_selector = xlingqg.answer_selection.AnswerSelector()

    text = ''

    while text != 'quit':
        text = input("Type a sentence or quit: ")
        doc = preprocessor.preprocess(text)
            
        answers = answer_selector.select_answers(doc)

        sentence_qas = []

        for sentence_with_answers in answers:
            for answer_indices in sentence_with_answers[1]:
                sentence = sentence_with_answers[0].text
                question = qg.generate_cross_lingual_question(sentence, answer_indices)
                answer = qg.generate_translated_answer(sentence,answer_indices)
                answer =  ' '.join(dict['token'] for dict in answer)
                result = {
                    "sentence": sentence,
                    "question":question,
                    "answer": answer
                }
                sentence_qas.append(result) 


if __name__ == '__main__':
    main()