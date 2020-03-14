import xlingqg_server.translation
import pytest 

def test_translate(translator):
    source_sentence = u'Federer entered the top 100 ranking for the first time on 20 September 1999.'
    translation = translator.translate_sentence(source_sentence)
    expected_translation = u'Federer trat zum ersten Mal am 20. September 1999 in die Top 100 ein.'

    assert translation == expected_translation

def test_translate_with_align(translator):    
    source_sentence = u'John does not live here.'
    translation_and_alignment = translator.translate_sentence_with_alignemnts(source_sentence)
    translation = translation_and_alignment[0]
    alignment = translation_and_alignment[1]
    source_tokens = translation_and_alignment[2]
    target_tokens = translation_and_alignment[3]
   
    expected_translation = u'John lebt hier nicht.'
    assert translation == expected_translation

    expected_alignment = [(0, 0), (3, 1), (4, 2), (2, 3), (5, 4)]
    assert alignment == expected_alignment

    expeceted_src_tokens = ['John', 'does', 'not', 'live', 'here', '.']
    assert source_tokens == expeceted_src_tokens

    expeceted_tgt_tokens = ['John', 'lebt', 'hier', 'nicht', '.']
    assert target_tokens == expeceted_tgt_tokens


