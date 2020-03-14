import pytest
import xlingqg.cross_lingual_qg

def test_encode_answer():
    answer_encoder = xlingqg.cross_lingual_qg.AnswerEncoder()
    encoded_tokens = answer_encoder.encode_answer([u'This', u'is', u'a', u'test', u'.'], [2,3])
    assert encoded_tokens == [u'This￨0', u'is￨0', u'a￨1', u'test￨1', u'.￨0' ]

def test_generate_xling_question(cross_lingual_qg):
    generated_question = cross_lingual_qg.generate_cross_lingual_question('John does not live here.', [0])
    assert generated_question == 'Wer lebt hier nicht?'

def test_generate_translated_answer_first_word(cross_lingual_qg):
    answer_tokens = cross_lingual_qg.generate_translated_answer('John does not live here.', [0])
    assert answer_tokens == [{'token':'John', 'index':0}]

def test_generate_translated_answer_later_word(cross_lingual_qg):
    answer_tokens = cross_lingual_qg.generate_translated_answer('John does not live here.', [4])
    assert answer_tokens == [{'token':'hier', 'index':2}]    
    
def test_generate_translated_answer_two_words(cross_lingual_qg):
    answer_tokens = cross_lingual_qg.generate_translated_answer('John does not live here.', [2,3])
    assert answer_tokens == [ {'token':'lebt', 'index':1}, {'token':'nicht', 'index':3}]        