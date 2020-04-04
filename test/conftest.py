import pytest 

@pytest.fixture
def translator():
    import xlingqg.translation
    return xlingqg.translation.Translator()

@pytest.fixture
def question_generator():
    import xlingqg.question_generation
    return xlingqg.question_generation.QuestionGenerator()    

@pytest.fixture
def cross_lingual_qg():
    import xlingqg.cross_lingual_qg
    return xlingqg.cross_lingual_qg.CrossLingualQuestionGenerator()

@pytest.fixture
def preprocessor():
    import xlingqg.preprocessing
    return xlingqg.preprocessing.Prepocessor()    