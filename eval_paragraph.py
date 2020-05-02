import xlingqg
import spacy 
import json
from tqdm import tqdm

def load_data(filename="./example_data/simplewiki.test.json"):
    with open(filename) as f:
        return json.load(f)

def write_to_file(data):
    with open('./example_data/simplewiki.test.gen.question.json', 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
def main():
    print_stats = True
    do_eval = True

    paragraphs = load_data()

    num_questions = 0
   
    if do_eval == True:    
        qg = xlingqg.CrossLingualQuestionGenerator()
        preprocessor = xlingqg.Prepocessor()
        answer_selector = xlingqg.answer_selection.AnswerSelector()

        paras_with_qas=[]    
        for para in tqdm(paragraphs):    
            text = para['text']
            name = para['name']

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
                    num_questions = num_questions + 1
            para_result = {
                "name":name,
                "paragraph":text,
                "sqas":sentence_qas
            }
            paras_with_qas.append(para_result)

        write_to_file(paras_with_qas)

    if print_stats == True:
        get_stats(paragraphs, num_questions)



    

def get_stats(paragraphs,num_questions):
    total_sents = 0
    total_tokens = 0
    total_paras = 20

    nlp = spacy.load('en_core_web_lg')

    for para in paragraphs:
        text = para['text']
        doc = nlp(text)

        for sent in doc.sents:
            total_sents = total_sents + 1

        for tok in doc:
            total_tokens = total_tokens + 1

    q_per_sent = num_questions/total_sents
    q_per_tok = num_questions/total_tokens
    print('Stats: Total Sentences: {}, Total Tokens: {}, Total_Questions: {}, Question_per_sentence = {}, Question_per_token = {}'.format(
        total_sents,total_tokens,num_questions,q_per_sent,q_per_tok))

if __name__ == '__main__':
    main()