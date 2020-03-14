import onmt.translate.translator
import xlingqg.model_loading


class QuestionGenerator():
    def __init__(self):
        config_parser = xlingqg.ConfigParser()
        config_parser.read_config(xlingqg.PATH_TO_CONFIG)  
        model_builder = xlingqg.OnmtModelBuilder()  

        self.model_config = config_parser.question_generation_config
        self.onmt_model = model_builder.build_model(self.model_config)      

    """Generates a question for the source_sentence
    
    Returns:
        [string] -- the generated question
    """
    def generate_question(self,source_sentece):
        questions = self.onmt_model.translate(
            src=[source_sentece],
            src_dir=None,
            batch_size = self.model_config.batch_size)
        return questions[1][0]