import onmt.translate.translator
import onmt.utils.parse as parse
import onmt.opts
import json
import os.path
import codecs
import fairseq.models.transformer


class OnmtModelConfig(object):

    def __init__(self, beam_size, path, replace_unknown, batch_size):
        self.path = path
        self.beam_size = beam_size
        self.replace_unknown = replace_unknown
        self.batch_size = batch_size


class FairseqModelConfig(object):
    def __init__(self, beam_size, path, checkpoint_filepath, bpe, bpe_codes, path_to_data):
        self.path = path
        self.beam_size = beam_size
        self.checkpoint_filepath = checkpoint_filepath
        self.bpe = bpe
        self.bpe_codes = bpe_codes
        self.path_to_data = path_to_data


class ConfigParser(object):

    def __init__(self):
        self.translation_config = None
        self.question_generation_config = None

    def read_config(self, path_to_config):
        with open(path_to_config) as config_file:
            config = json.load(config_file)
            self._parse_config(config)

    def _parse_config(self, config):
        model_root_dir = config['model_root_dir']
        self.translation_config = self._parse_fairseq_model_config(
            config['translation_model'], model_root_dir)
        self.question_generation_config = self._parse_onmt_model_config(
            config['qg_model'], model_root_dir)

    def _parse_onmt_model_config(self, model_config_opts, model_root_dir):
        model_beam_size = model_config_opts['beam_size']
        model_path = os.path.join(model_root_dir,  model_config_opts['model'])
        batch_size = model_config_opts['batch_size']
        replace_unknown = model_config_opts['replace_unknown']
        model_config = OnmtModelConfig(
            beam_size=model_beam_size, path=model_path, replace_unknown=replace_unknown, batch_size=batch_size)
        return model_config

    def _parse_fairseq_model_config(self, model_config_opts, model_root_dir):
        model_beam_size = model_config_opts['beam_size']
        checkpoint_filepath = model_config_opts['model']
        data_path = model_config_opts['path_to_data']
        bpe = model_config_opts['bpe']
        bpe_codes = model_config_opts['bpe_codes']
    

        model_config = FairseqModelConfig(beam_size=model_beam_size, path=model_root_dir, checkpoint_filepath=checkpoint_filepath,
                                          bpe=bpe, bpe_codes=bpe_codes, path_to_data=data_path)
        return model_config


class FairseqModelBuilder(object):
    def build_model(self, model_config):
        model = fairseq.models.transformer.TransformerModel.from_pretrained(
            model_name_or_path=model_config.path,
            checkpoint_file=model_config.checkpoint_filepath,
            data_name_or_path=model_config.path_to_data,
            bpe=model_config.bpe,
            bpe_codes=model_config.bpe_codes
        )
        return model


class OnmtModelBuilder(object):

    def build_model(self, model_config):
        model_args = self._build_model_args(model_config)
        model = onmt.translate.translator.build_translator(
            model_args, report_score=False)
        return model

    def _build_model_args(self, model_config):
        parser = self._get_parser()
        args = []
        args.append('-model')
        args.append(str(model_config.path))
        args.append('-beam_size')
        args.append(str(model_config.beam_size))
        args.append('-src')
        args.append('dummy_src')
        if model_config.replace_unknown: args.append('--replace_unk') 
        args.append('-phrase_table')
        args.append('./models/phrase_table.txt')
        model_args = parser.parse_args(args)
        return model_args

    def _get_parser(self):
        parser = parse.ArgumentParser(description='translate.py')
        onmt.opts.translate_opts(parser)
        return parser


def main():
    path_to_config_file = './models/config.json'
    config_parser = ConfigParser()
    config_parser.read_config(path_to_config_file)

    model_builder = OnmtModelBuilder()
    onmt_model = model_builder.build_model(
        config_parser.question_generation_config)
    translations = onmt_model.translate(
        src=[u'Roger￨1 Federer￨1 was￨0 born￨0 <unk>￨0 in￨0 Switzerland￨0'],
        src_dir=None,
        batch_size = 30
    )
    test = translations[1]
    print(test)

    fairseq_builder = FairseqModelBuilder()
    fairseq_model = fairseq_builder.build_model(config_parser.translation_config)
    input = fairseq_model.encode(test[0][0])
    args = {'print_alignment':True,
            'tokenizer':'moses'}
    hypos = fairseq_model.generate(input, beam=5, verbose=False,**args)
    for hypo in hypos:
        translation = fairseq_model.decode(hypo['tokens'])
        print(translation)
        print(hypo['alignment'])

if __name__ == '__main__':
    main()
