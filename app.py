import xlingqg
from flask import Flask, jsonify, request
import waitress
import argparse


def run(host, port):
    app = Flask(__name__)
    qg = xlingqg.CrossLingualQuestionGenerator()

    @app.route("/Question", methods=['POST'])
    def generate_question():
        # Roger Federer was born 1986 in Switzerland.
        request_payload = request.get_json(force=True)
        source_sentence = request_payload['source']
        answer_positions = request_payload['answerPositions']
        question = qg.generate_cross_lingual_question(source_sentence, answer_positions)
        answer = qg.generate_translated_answer(source_sentence, answer_positions)
        response = {"question": question,
                    "answer": ' '.join(dict['token'] for dict in answer)
                    }
        return jsonify(response)

    waitress.serve(app, host=args.host, port=args.port)


def main():
    qg = xlingqg.CrossLingualQuestionGenerator()
    question = qg.generate_cross_lingual_question(
        u'Roger Federer was born 1986 in Switzerland.', [5, 6])
    answer = qg.generate_translated_answer(
        u'Roger Federer was born 1986 in Switzerland.', [5, 6])

    print('{} {}'.format(question,
                         ' '.join(dict['token'] for dict in answer)))

    question = qg.generate_cross_lingual_question(
        u'Roger Federer was born 1986 in Switzerland.', [0, 1])
    answer = qg.generate_translated_answer(
        u'Roger Federer was born 1986 in Switzerland.', [0, 1])

    print('{} {}'.format(question,
                         ' '.join(dict['token'] for dict in answer)))

    question = qg.generate_cross_lingual_question(
        u'Roger Federer was born 1986 in Switzerland.', [4])
    answer = qg.generate_translated_answer(
        u'Roger Federer was born 1986 in Switzerland.', [4])

    print('{} {}'.format(question,
                         ' '.join(dict['token'] for dict in answer)))


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    return parser


if __name__ == '__main__':
    # main()
    parser = _get_parser()
    args = parser.parse_args()
    run(host=args.host, port=args.port)
