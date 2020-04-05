import xlingqg
import spacy 
import json
from tqdm import tqdm

def load_data(filename="./example_data/squad.test.json"):
    with open(filename) as f:
        return json.load(f)

def select_paragraphs(paras):
    selected_contexts = []
    randomly_selected_paragraph_ids = [('00','07'),('05','00'),('05','06'), ('05','26'), ('07','34'),
                                       ('07','59') , ('09','10'), ('09','22'), ('13', '00') ,('13','04'),
                                       ('19','001'), ('19','113'), ('19','122'), ('23','07'), ('23','37'),
                                       ('30', '25'), ('30','13'), ('37', '05'), ('38','13'), ('42','06') ]
    for paragraph_id in randomly_selected_paragraph_ids:
        paragraph = paras[int(paragraph_id[0])]
        context = paragraph['paragraphs'][int(paragraph_id[1])]['context']
        selected_contexts.append(context)

    return selected_contexts

def write_to_file(data):
    with open('./example_data/squad.test.gen.question.json', 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
def main():
    print_stats = False
    
    paragraphs = load_data()
    selected_paragraphs = select_paragraphs(paragraphs)
   
    if print_stats == True:
        get_stats(selected_paragraphs)
        
    qg = xlingqg.CrossLingualQuestionGenerator()
    preprocessor = xlingqg.Prepocessor()
    answer_selector = xlingqg.answer_selection.AnswerSelector()

    paras_with_qas=[]    
    for para in tqdm(selected_paragraphs[:1]):    
        doc = preprocessor.preprocess(para)
        answers = answer_selector.select_answers(doc)

        sentence_qas = []

        for sentence_with_answers in answers[:1]:
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
        para_result = {
            "paragraph":para,
            "sqas":sentence_qas
        }
        paras_with_qas.append(para_result)

    write_to_file(paras_with_qas)



    

def get_stats(selected_paragraphs):
    total_sents = 0
    total_tokens = 0
    total_paras = 20
    total_question = 0

    nlp = spacy.load('en_core_web_lg')

    for para in selected_paragraphs:
        doc = nlp(para)

        for sent in doc.sents:
            total_sents = total_sents + 1

        for tok in doc:
            total_tokens = total_tokens + 1

    q_per_sent = total_question/total_sents
    q_per_tok = total_question/total_tokens
    print('Stats: Total Sentences: {}, Total Tokens: {}, Total_Questions: {}, Question_per_sentence = {}, Question_per_token = {}'.format(
        total_sents,total_tokens,total_question,q_per_sent,q_per_tok))

if __name__ == '__main__':
    main()