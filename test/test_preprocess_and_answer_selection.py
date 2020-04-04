import xlingqg.preprocessing
import xlingqg.answer_selection
import pytest

def test_answer_selection(preprocessor):
    text = """Roger Federer is a Swiss professional tennis player.
    He was born on 1 January 1992 in Switzerland.
    He has won 20 Grand Slam titles.
    Roger Federer said it was nice winning them.
    20 Grand Slam titles were won by him.
    He won the titles because he is the best player."""
    doc = preprocessor.preprocess(text)
    answer_selector = xlingqg.answer_selection.AnswerSelector()
    answers = answer_selector.select_answers(doc)
    print(answers)

