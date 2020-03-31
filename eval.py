import xlingqg
import sacrebleu
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm


def generate_questsions(sentence_list):
    qg= xlingqg.QuestionGenerator()
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
        translations.append(translator.translator.tokenize(translation))   
    return translations

def evaluate(hypos, references):
    bleu =  sacrebleu.corpus_bleu(hypos,references)   
    return bleu

def load_file_lines(filepath):
    with open(filepath) as f:
        return f.read().splitlines()        

def evaluate_translation(translate_src, translate_ref):
    refs = [load_file_lines(translate_ref)]
    hypos = do_translations(load_file_lines(translate_src)) 
    return evaluate(hypos,refs)    

def evaluate_qg(qg_src, qg_ref):
    hypos = generate_questsions(load_file_lines(qg_src)) 
    refs = [ load_file_lines(qg_ref) ]
    return evaluate(hypos,refs)          


def main(eval_qg, eval_translate, qg_src, qg_ref, translate_src,translate_ref):
    if eval_qg: 
        print("QG-Score: {}".format(evaluate_qg(qg_src,qg_ref).format()))
    if eval_translate:
        print("Translate-Score: {}".format(evaluate_translation(translate_src,translate_ref).format()))    

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qg", action="store_true",default=False)
    parser.add_argument("--translate", action="store_true", default=True )
    parser.add_argument("--translate_src", type=str, default="./example_data/translate.test.en")
    parser.add_argument("--translate_ref", type=str, default="./example_data/translate.test.de")
    parser.add_argument("--qg_src", type=str, default="./example_data/qg.test.sentence")
    parser.add_argument("--qg_ref", type=str, default="./example_data/qg.test.question")
    return parser

if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    main(args.qg, args.translate, args.qg_src, args.qg_ref, args.translate_src,args.translate_ref)