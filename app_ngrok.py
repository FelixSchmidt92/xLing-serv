import xlingqg
from flask import Flask, jsonify, request
import waitress
import argparse
from flask_ngrok import run_with_ngrok


def run(host, port):
    app = Flask(__name__)
    run_with_ngrok(app)   
    qg = xlingqg.CrossLingualQuestionGenerator()
    preprocessor = xlingqg.Prepocessor()
    answer_selector = xlingqg.answer_selection.AnswerSelector()

    @app.route("/Question", methods=['POST'])
    def generate_question():     
        request_payload = request.get_json(force=True)
        text = request_payload['text']

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

        return jsonify(sentence_qas)

    waitress.serve(app, host=args.host, port=args.port)

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    run(host=args.host, port=args.port)
