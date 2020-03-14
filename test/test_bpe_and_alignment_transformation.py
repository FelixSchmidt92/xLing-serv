import xlingqg_server.translation
import pytest

def test_det_word_to_subword():
    a = xlingqg_server.translation.determine_word_to_subword_ranges(['This', 'is', 'a', 'sub@@'
        , 'wo@@', 'rd', 'te@@', 'st@@', 'case', '.'])

    assert a == [(0,0),(1,1),(2,2),(3,5),(6,8),(9,9)]

def test_det_word_to_subword_begin():
    a = xlingqg_server.translation.determine_word_to_subword_ranges(['Th@@', 'is', 'a', 'nice', 'test', '.'])
    assert a == [(0,1),(2,2),(3,3),(4,4),(5,5)]    

def test_det_word_to_subword_one_word():
    a = xlingqg_server.translation.determine_word_to_subword_ranges(['This'])
    assert a == [(0,0)]        

def test_det_word_to_subword_one_subword():
    a = xlingqg_server.translation.determine_word_to_subword_ranges(['Th@@', 'is'])
    assert a == [(0,1)]   

def test_det_word_to_subword_initial():
    a = xlingqg_server.translation.determine_word_to_subword_ranges([])
    assert a == [] 

def test_det_subword_to_word():
    a = xlingqg_server.translation.determine_subword_to_word(['This', 'is', 'a', 'sub@@'
        , 'wo@@', 'rd', 'te@@', 'st@@', 'case', '.'])

    assert a == [0,1,2,3,3,3,4,4,4,5]

def test_det_subword_to_word_start_subword():
    a = xlingqg_server.translation.determine_subword_to_word(['Th@@', 'is', 'is', 'a', 'test', '.'])

    assert a == [0,0,1,2,3,4]    

def test_det_subword_to_word_initial():
    a = xlingqg_server.translation.determine_subword_to_word([])
    assert a == []     

def test_subword_align_to_word_align():
    source_subwords = ['Feder@@', 'er', 'is', 'a', 'go@@', 'od', 'man', '.']
    target_subwords = ['Ein', 'gu@@', 'ter', 'Ma@@', 'nn', 'ist', 'Federer', '.']
    subword_alignment = [(0,6), (1,6), (2,5), (3,0), (4,1), (5,2),(6,3),(6,4),(7,7)]

    alignment = xlingqg_server.translation.subword_align_to_word_align(source_subwords,target_subwords,subword_alignment)
    
    expected_alignment = [(0,4),(1,3),(2,0),(3,1),(4,2),(5,5)]
    assert alignment == expected_alignment
    

