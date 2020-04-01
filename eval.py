import xlingqg
import sacrebleu
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm


def generate_questsions(sentence_list):
    qg = xlingqg.QuestionGenerator()
    detokenizer = MosesDetokenizer(lang='en')
    generated_questions = []
    for sentence in tqdm(sentence_list):
        question = qg.generate_question(sentence)
        generated_questions.append(detokenizer.detokenize(question))

    return generated_questions


def do_translations(sentence_list):
    translator = xlingqg.Translator()
    translations = []
    sentence_list = sentence_list
    for sentence in tqdm(sentence_list):
        sentence_without_bpe = translator.translator.remove_bpe(sentence)
        translation = translator.translate_sentence(sentence_without_bpe)
        translation = translator.translator.tokenize(translation)
        translations.append(translation)
    return translations

def evaluate(hypos, references):
    bleu = sacrebleu.corpus_bleu(hypos, references)
    return bleu

def load_file_lines(filepath):
    with open(filepath) as f:
        return f.read().splitlines()[:2]

def write_result(filepath,results):
    with open(filepath, 'w') as f:
        f.write("\n".join(str(result) for result in results))   

def evaluate_translation(translate_src, translate_ref,translate_result):
    refs = [load_file_lines(translate_ref)]
    hypos = do_translations(load_file_lines(translate_src))
    write_result(translate_result,hypos)
    return evaluate(hypos, refs)

def evaluate_qg(qg_src, qg_ref,qg_result):
    detokenizer = MosesDetokenizer()
    hypos = generate_questsions(load_file_lines(qg_src))
    refs = load_file_lines(qg_ref)
    refs_detok = []
    for ref in refs: 
       refs_detok.append( detokenizer.detokenize([ref]))
    write_result(qg_result,hypos)   
    return evaluate(hypos, [refs_detok])


def main(eval_qg, eval_translate, qg_src, qg_ref, translate_src, 
        translate_ref, translate_result, qg_result):
    if eval_qg:
        qg_bleu = evaluate_qg(qg_src, qg_ref,qg_result)
        print("QG-Score: {}".format(qg_bleu.format()))
    if eval_translate:
        translation_bleu = evaluate_translation(translate_src, translate_ref, translate_result)
        print("Translate-Score: {}".format(translation_bleu.format()))


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg", action="store_true", default=False)
    parser.add_argument("--translate", action="store_true", default=False)
    parser.add_argument("--translate_src", type=str,
                        default="./example_data/translate.test.en")
    parser.add_argument("--translate_ref", type=str,
                        default="./example_data/translate.test.de")
    parser.add_argument("--qg_src", type=str,
                        default="./example_data/qg.test.sentence")
    parser.add_argument("--qg_ref", type=str,
                        default="./example_data/qg.test.question")
    parser.add_argument("--qg_result", type=str,
                        default="./example_data/qg.pred.question")
    parser.add_argument("--translate_result", type=str,
                        default="./example_data/translate.pred.de")
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    main(eval_qg=args.qg, eval_translate=args.translate, qg_src=args.qg_src,
         qg_ref=args.qg_ref, translate_src=args.translate_src,
         translate_ref=args.translate_ref, translate_result=args.translate_result, qg_result=args.qg_result)
