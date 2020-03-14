import xlingqg_server.question_generation
import xlingqg_server.model_loading
import pytest


def test_generate_question(question_generator):
    source_sentence = u'Roger￨0 Federer￨ was￨0 born￨0 1981￨0 in￨1 Switzerland￨1 .￨0'

    generated_question = question_generator.generate_question(source_sentence)
    assert generated_question == ['Where was Roger Federer born ?']