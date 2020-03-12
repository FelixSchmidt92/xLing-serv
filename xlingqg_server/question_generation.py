import fairseq.models.transformer
import onmt.translate.translator
import re

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
    [list] -- A list of word alignments in pharaoh format
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

    return word_align

