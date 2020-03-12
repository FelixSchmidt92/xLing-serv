import xlingqg_server.model_loading
import fairseq.models.transformer
import onmt.translate.translator
import re
         
def bpe_ranges(bpe_tokens):
    pattern = re.compile('.*@@$')
    begin_of_word = 0
    subword_to_word = []
    for index, token in enumerate(bpe_tokens):
        if not pattern.match(token):
            subword_to_word.append((begin_of_word,index))
            begin_of_word = index + 1
    return subword_to_word
