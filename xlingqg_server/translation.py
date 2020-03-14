import fairseq.models.transformer
import xlingqg_server.model_loading
import re


class Translator():
    def __init__(self):
        config_parser = xlingqg_server.ConfigParser()
        config_parser.read_config(xlingqg_server.PATH_TO_CONFIG)  
        model_builder = xlingqg_server.FairseqModelBuilder()  

        self.model_config = config_parser.translation_config
        self.translator = model_builder.build_model(self.model_config)      

    """Translates the source sentence with alignments
    
    Returns:
        string -- the translation of the sentence
    """
    def translate_sentence(self,source_sentece):
        input = self.translator.encode(source_sentece)
        args = {'tokenizer':'moses'}
        hypos = self.translator.generate(input, beam=5, verbose=False,**args)
        hypo = hypos[0]    

        return self.translator.decode(hypo['tokens'])  

    """Translates the source sentence with word level alignments between source and
    translation
    
    Returns:
        [tupel] -- the translation of the source sentence, the word-level alignment
        between source and translation, the source tokens and the translation tokens
    """
    def translate_sentence_with_alignemnts(self,source_sentence):
        input = self.translator.encode(source_sentence)
        args = {'tokenizer':'moses',
            'print_alignment':True,}
        hypos = self.translator.generate(input, beam=5, verbose=False,**args)
        hypo = hypos[0]    

        translation = self.translator.decode(hypo['tokens'])  

        trans_bpe_tokens = self.translator.string(hypo['tokens']).split()
        source_bpe_tokens = self.translator.string(input).split()
        subword_alignments = hypo['alignment']

        word_alignments = subword_align_to_word_align(source_bpe_tokens, trans_bpe_tokens, subword_alignments)
        return (translation, word_alignments, self.translator.tokenize(source_sentence).split(), self.translator.tokenize(translation).split())

""" Creates a list of tuples containing the index of first subword token of the word
    and the index of the last subword token of the word
Returns:
    [list] -- At the i-th postion the list contains a tuple with the index of the 
              first subword and the index of the last subword corresponding to the i-th word   
"""
def determine_word_to_subword_ranges(bpe_tokens):
    pattern = re.compile('.*@@$')
    begin_of_word = 0
    word_to_subword = []
    for index, token in enumerate(bpe_tokens):
        if not pattern.match(token):
            word_to_subword.append((begin_of_word,index))
            begin_of_word = index + 1
    return word_to_subword


""" Creates a list with the indicies of the subwords and the 
Returns:
    [list] -- At the i-th postion the list contains the index of word
              corresponding to the i-th subword           
"""
def determine_subword_to_word(bpe_tokens):
    pattern = re.compile('.*@@$')
    corresponding_word_index = 0
    subword_to_word = []
    for index, token in enumerate(bpe_tokens):
        subword_to_word.append(corresponding_word_index) 
        if not pattern.match(token):
            corresponding_word_index = corresponding_word_index + 1   
    return subword_to_word

""" Constructs word alignments from the subword alignments. If a subword
    of the source sentence has an alignment to a subword of the target sentence 
    the corresponding words are counted as aligned as well.

Returns:
    [set] -- A list of word alignments in pharaoh format
"""
def subword_align_to_word_align(source_subwords,target_subwords,subword_align):

    word_align = []

    source_subword_to_word = determine_subword_to_word(source_subwords)
    target_subword_to_word = determine_subword_to_word(target_subwords)

    for align in subword_align:
        source_subword_index = align[0]
        target_subword_index = align[1]
        
        source_word_index = source_subword_to_word[source_subword_index]
        target_word_index = target_subword_to_word[target_subword_index]

        word_align.append((source_word_index,target_word_index))

    word_align_no_duplicate = dict.fromkeys(word_align)
    return list(word_align_no_duplicate)

