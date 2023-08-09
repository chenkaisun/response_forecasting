from torch import optim
import torch
import os
from transformers import AutoTokenizer, AutoModel, BertModel,BertTokenizerFast, AutoModelForSequenceClassification, BartTokenizer, BertForQuestionAnswering, BartModel
from transformers import EncoderDecoderModel, AutoModelForCausalLM, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderConfig, AutoConfig, \
    BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel, AutoModelWithLMHead
# from .cse import EventReasoningEncoder
from .baseline import BaselineEncoder
from .bart.modeling_bart import SEEBartForConditionalGeneration, BartEncoder
from model.fet import FET
from torch.nn import DataParallel
from IPython import embed
from transformers import AutoTokenizer, AutoModelWithLMHead

from transformers import EncoderDecoderModel, AutoModelForCausalLM, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderConfig, AutoConfig, \
    BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel, AutoModel, OpenAIGPTLMHeadModel
# import wandb
def get_model(args, tokenizer, **kwargs):
    model=None

    # if args.use_event_tag:
    #     special_tokens_dict = {'additional_special_tokens': [f'[EVT{i}]' for i in range(100)]}
    #     tokenizer.add_special_tokens(special_tokens_dict)
    revision_id = "ea0107eec489da9597e9eefd095eb691fcc7b4f9" if "bart-base" in args.plm else "9e1698384a6941529439cc19e276da61ba6cfa26"
    print("model_name", args.model_name)
    print("plm", args.plm)
    if args.model_name == "ours_clf":
        model = FET(args)
        # id2label={i:str(i) for i in range(args.out_dim)}
        # label2id={str(i):i for i in range(args.out_dim)}
        # model = AutoModelForSequenceClassification.from_pretrained(args.plm, num_labels=args.out_dim, id2label=id2label, label2id=label2id)


        # model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.plm, args.plm)
        # encoder_config = AutoConfig.from_pretrained(args.plm)
        # print("encoder_config",encoder_config)
        # encoder = BaselineEncoder.from_pretrained(args.plm) #,, config=encoder_config bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id
        # print("encoder.config",encoder.config)
        # # decoder = AutoModelForCausalLM.from_pretrained(args.plm, add_cross_attention=True, is_decoder=True)
        # # ed_config=EncoderDecoderConfig.from_encoder_decoder_configs()
        # # model = EncoderDecoderModel(encoder=encoder, decoder=decoder ) #, tie_encoder_decoder=True decoder_model=decoder,
        # model = EncoderDecoderModel.from_encoder_decoder_pretrained(decoder_pretrained_model_name_or_path=args.plm, encoder_model=encoder, encoder_config=encoder_config, tie_encoder_decoder=True )
        # if args.task_mode in ["gen",'prt'] :
        #
        #     encoder = BertGenerationEncoder.from_pretrained(args.plm, bos_token_id=101, eos_token_id=102) # bert large uncased assumed
        #     # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
        #     decoder = BertGenerationDecoder.from_pretrained(args.plm, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
        #     model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        #
        # elif args.task_mode == "clf":
        #     model = FET(args,tokenizer)


    elif args.model_name == "bart":
        print("args.model_name", args.model_name)
        if args.task_mode in ["gen",'prt'] :
            if "bart-tiny" in args.plm:
                model=BartForConditionalGeneration.from_pretrained(args.plm)
            else:
                model=BartForConditionalGeneration.from_pretrained(args.plm, revision=revision_id)
            print("revision")

        elif args.task_mode == "clf":
            model = FET(args,tokenizer)

        # decoder default plm
    elif args.model_name == "t5":
        model=T5ForConditionalGeneration.from_pretrained(args.plm)
        # decoder default plm
    # elif args.model_name == "gpt2":
    #     model = GPT2LMHeadModel.from_pretrained(args.plm)
    elif "gpt" in args.model_name:
        model = AutoModelWithLMHead.from_pretrained(args.plm)
    elif args.model_name == "predictor":

        if "t5" in args.plm:
            model = AutoModelWithLMHead.from_pretrained(args.plm)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.plm)

        # encoder = EventReasoningEncoder.from_pretrained(args.plm,  args=args, pretrained_concept_emb=kwargs['pretrained_concept_emb']) #, bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id
        # encoder.add_modules(args, **kwargs)
        # if args.use_event_tag:
        #     encoder.resize_token_embeddings(len(tokenizer))
        #
        # # print("\n\nencoder", encoder)
        #
        # # decoder = BertGenerationDecoder.from_pretrained(args.plm, add_cross_attention=True, is_decoder=True)
        # decoder = AutoModelForCausalLM.from_pretrained(args.plm, add_cross_attention=True, is_decoder=True)
        # # print("\n\ndecoder", decoder)
        #
        # # bos_token_id = tokenizer.cls_token_id, eos_token_id = tokenizer.sep_token_id
        # model = EncoderDecoderModel(encoder=encoder, decoder=decoder ) #, tie_encoder_decoder=True
        # print("model.config",model.config)
        # print(bert2bert.config)
    else:
        print("model not implemented")
        raise NotImplementedError
    print(model)


    if "get_input_embeddings" in dir(model):
        if model.get_input_embeddings().weight.shape[0]!=len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))


    if args.task_mode in ["gen",'prt'] :


        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

        bos_token_id=bos_token_id if bos_token_id is not None else tokenizer.pad_token_id # t5 requires pad

        model.config.bos_token_id = bos_token_id
        model.config.eos_token_id = eos_token_id

        model.config.decoder_start_token_id = bos_token_id
        # print("model.config.decoder_start_token_id",model.config.decoder_start_token_id)
        # embed()
        model.config.num_beams = args.num_beams
        model.config.max_length = args.generated_max_length  # 142
        model.config.min_length = args.generated_min_length
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = args.length_penalty
        model.config.early_stopping = True
        model.config.top_p = 0.92

        if "bert" in args.model_name:
            model.config.decoder_start_token_id = tokenizer.cls_token_id
            model.config.eos_token_id = tokenizer.sep_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.bos_token_id = tokenizer.cls_token_id

            model.config.vocab_size = model.config.encoder.vocab_size

            model.config.max_length = args.generated_max_length  # 142
            model.config.min_length = args.generated_min_length
            model.config.no_repeat_ngram_size = 3
            model.config.early_stopping = True
            model.config.length_penalty = args.length_penalty
            model.config.num_beams = args.num_beams

            model.config.decoder.decoder_start_token_id = tokenizer.cls_token_id
            model.config.decoder.eos_token_id = tokenizer.sep_token_id
            model.config.decoder.pad_token_id = tokenizer.pad_token_id
            model.config.decoder.bos_token_id = tokenizer.cls_token_id
            model.config.decoder.max_length = args.generated_max_length  # 142
            model.config.decoder.min_length = args.generated_min_length
            model.config.decoder.no_repeat_ngram_size = 3
            model.config.decoder.early_stopping = True
            model.config.decoder.length_penalty = args.length_penalty
            model.config.decoder.num_beams = args.num_beams

        # print("model.config",model.config)
    return model


def load_model_from_path(model, optimizer, model_path, gpu=True, device=None):
    model_epoch, best_dev_score = 0, -float("inf")

    if os.path.isfile(model_path) and False:
        print("Loading saved model")

        model_dic=torch.load(model_path)
        model.load_state_dict(model_dic)
        #
        #
        # saved_model_info = torch.load(model_path)
        # model.load_state_dict(saved_model_info['model_state_dict'])
        # model_epoch = saved_model_info["epoch"]
        # best_dev_score = saved_model_info["best_dev_score"]
        # optimizer.load_state_dict(saved_model_info['optimizer_state_dict'])
        # if gpu:
        #     model = model.cuda()
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.cuda()

    else:
        # if not os.path.isfile(model_path):
        #     print("Saved model not found")
        if gpu:
            # model = DataParallel(model)
            model = model.to(device)

        # embed()
        # print("device cur", model.device)
        # print("device cur1", model.get_device())
    # model.to(device)
    return model, optimizer, model_epoch, best_dev_score
