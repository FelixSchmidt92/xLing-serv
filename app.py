import xlingqg_server

def main():
    path_to_config_file = './models/config.json'
    config_parser = xlingqg_server.ConfigParser()
    config_parser.read_config(path_to_config_file)

    model_builder = xlingqg_server.OnmtModelBuilder()
    onmt_model = model_builder.build_model(
        config_parser.question_generation_config)

    fairseq_builder = xlingqg_server.FairseqModelBuilder()
    fairseq_model = fairseq_builder.build_model(config_parser.translation_config)

    translations = onmt_model.translate(
        src=[u'Roger￨0 Federer￨ was￨0 born￨0 1981￨0 in￨1 Switzerland￨0'],
        src_dir=None,
        batch_size = 30
    )
    test = translations[1]
    print(test)

    input = fairseq_model.encode(test[0][0])
    args = {'print_alignment':True,
            'tokenizer':'moses'}
    hypos = fairseq_model.generate(input, beam=5, verbose=False,**args)
    for hypo in hypos:
        translation = fairseq_model.decode(hypo['tokens'])
        print(translation)
        print(hypo['alignment']) 


    test_source = u'Federer entered the top 100 ranking for the first time on 20 September 1999'
    input = fairseq_model.encode(test_source)
    print(test_source)
    args = {'print_alignment':True,
            'tokenizer':'moses'}
    hypos = fairseq_model.generate(input, beam=5, verbose=False,**args)

    hypo = hypos[0]    

    translation = fairseq_model.decode(hypo['tokens'])
    bpe = fairseq_model.tokenize(translation)
    print(fairseq_model.apply_bpe(test_source))
    print(fairseq_model.apply_bpe(translation))
    print(hypo['alignment'])   

    trans_bpe_tok = fairseq_model.string(hypo['tokens'])
    source_bpe_tok = fairseq_model.string(input)
    print(source_bpe_tok)
    print(trans_bpe_tok)
    source_tokens = source_bpe_tok.split()
    trans_tokens = trans_bpe_tok.split()

    for align in hypo['alignment']:
        print(source_tokens[align[0]] + ' -> ' + trans_tokens[align[1]])

if __name__ == '__main__':
    main()