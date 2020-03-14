import pytest 

@pytest.fixture
def translator():
    import xlingqg.translation
    return xlingqg.translation.Translator()

@pytest.fixture
def question_generator():
    import xlingqg.question_generation
    return xlingqg.question_generation.QuestionGenerator()    