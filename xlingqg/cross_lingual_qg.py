import xlingqg.translation
import xlingqg.question_generation
import sacremoses


class AnswerEncoder():

    """Encodes the answer in the source sentence with a binary indicator 
        0 - the token is not part of the answer
        1 - the token is part of the answer    
    Returns:
        list -- answer encoded tokens
    """

    def encode_answer(self, source_sentence_tokens, answer_positions):
        source_sentence_tokens_encoded = []
        for index, token in enumerate(source_sentence_tokens):
            if index in answer_positions:
                source_sentence_tokens_encoded.append(u'{0}￨1'.format(token))
            else:
                source_sentence_tokens_encoded.append(u'{0}￨0'.format(token))
        return source_sentence_tokens_encoded


class CrossLingualQuestionGenerator():

    def __init__(self):
        self.question_generator = xlingqg.QuestionGenerator()
        self.translator = xlingqg.Translator()
        self.tokenizer = sacremoses.MosesTokenizer()
        self.detokenizer = sacremoses.MosesDetokenizer()
        self.answer_encoder = AnswerEncoder()

    def generate_cross_lingual_question(self, source_sentence, answer_positions):
        source_tokens = self.tokenizer.tokenize(source_sentence)
        source_tokens_encoded = self.answer_encoder.encode_answer(
            source_tokens, answer_positions)

        source_sentence_encoded = self.detokenizer.detokenize(
            source_tokens_encoded)
        generated_question = self.question_generator.generate_question(
            source_sentence_encoded)[0]

        translated_question = self.translator.translate_sentence(
            generated_question)

        return translated_question

    """ Translates the sentence and uses the alignments between source sentence
    and translation to determine which tokens of the translation represent the 
    answer tokens of the source sentence
    
    Returns:
        list -- list of tuples containing the translated answer tokens and their positions
    """

    def generate_translated_answer(self, source_sentence, answer_positions):
        source_tokens = self.tokenizer.tokenize(source_sentence)

        translation_result = self.translator.translate_sentence_with_alignemnts(
            source_sentence)
        translation = translation_result[0]
        alignments = translation_result[1]
        source_tokens = translation_result[2]
        target_tokens = translation_result[3]

        translated_answer_indices = self.__get_translated_answer_indices(
            answer_positions, alignments)

        return self.__get_translated_answer_tokens(target_tokens, translated_answer_indices)

    """ Determines the answer tokens of the translation based on their indices.
    The list of answer tokens is sorted ascending by their position in the translation    
    Returns:
        list -- Contains the translated answer tokens
    """

    def __get_translated_answer_tokens(self, target_tokens, translated_answer_indices):
        answer_tokens = []
        for index, token in enumerate(target_tokens):
            if index in translated_answer_indices:
                answer_tokens.append({'token': token,
                                      'index': index})
        return answer_tokens

    """ Determines the indices of the answer tokens in the translated sentence
    based on the answer indices in the source sentence and the alignments between
    the tokens of the source sentence and the tokens of the translation
    
    Returns:
        list -- the inidices of the answer tokens in the translation
    """

    def __get_translated_answer_indices(self, answer_positions, alignments):
        translated_answer_indices = []

        for answer_index in answer_positions:
            for align in alignments:
                if(align[0]) == answer_index:
                    translated_answer_indices.append(align[1])

        return list(dict.fromkeys(translated_answer_indices))
