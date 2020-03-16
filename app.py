import xlingqg
from flask import Flask
import waitress
import argparse

app = Flask(__name__)

@app.route("/")
def hello():
    return "hi"

def main():
    qg = xlingqg.CrossLingualQuestionGenerator()
    question = qg.generate_cross_lingual_question(
        u'Roger Federer was born 1986 in Switzerland.', [5, 6])
    answer = qg.generate_translated_answer(
        u'Roger Federer was born 1986 in Switzerland.',[5,6])

    print('{} {}'.format(question,
                         ' '.join(dict['token'] for dict in answer)))

    question = qg.generate_cross_lingual_question(
        u'Roger Federer was born 1986 in Switzerland.', [0, 1])
    answer = qg.generate_translated_answer(
        u'Roger Federer was born 1986 in Switzerland.',[0,1])

    print('{} {}'.format(question,
                         ' '.join(dict['token'] for dict in answer)))

    question = qg.generate_cross_lingual_question(
        u'Roger Federer was born 1986 in Switzerland.', [4])
    answer = qg.generate_translated_answer(
        u'Roger Federer was born 1986 in Switzerland.',[4])

    print('{} {}'.format(question,
                         ' '.join(dict['token'] for dict in answer)))                         

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="5000")
    parser.add_argument("--url_root", type=str, default="/questionGeneration")
    return parser

if __name__ == '__main__':
    main()
    parser = _get_parser()
    args = parser.parse_args()
    print(args.url_root)
    waitress.serve(app, host=args.host, port=args.port)

