import pytest 

@pytest.fixture
def translator():
    import xlingqg_server.translation
    return xlingqg_server.translation.Translator()

@pytest.fixture
def question_generator():
    import xlingqg_server.question_generation
    return xlingqg_server.question_generation.QuestionGenerator()    