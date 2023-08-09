import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn import Linear
from transformers import AutoModel, AutoModelForSequenceClassification
from model.model_utils import get_tensor_info

# from model.gnn import *
# from train_utils import get_tensor_info


# from torchtext.vocab import GloVe


class FET(torch.nn.Module):
    def __init__(self, args):
        super(FET, self).__init__()

        self.components = args.components
        self.num_labels = args.out_dim
        id2label={i:str(i) for i in range(self.num_labels)}
        label2id={str(i):i for i in range(self.num_labels)}
        # self.plm = AutoModelForSequenceClassification.from_pretrained(args.plm, num_labels=args.out_dim, id2label=id2label, label2id=label2id)
        self.plm = AutoModel.from_pretrained(args.plm)

        self.combiner = Linear(args.plm_hidden_dim, args.out_dim)

        self.dropout = args.dropout

        # self.loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        # self.loss = nn.BCEWithLogitsLoss()

        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                masked_input_ids=None,
                masked_texts_attention_mask=None,
                ids=None,
                tweet_ids=None,
                in_train=None,
                print_output=False):
        final_vec = []

        "=========Original Text Encoder=========="
        # embed()
        # print("texts", get_tensor_info(texts))
        # print("texts_attn_mask", get_tensor_info(texts_attn_mask))
        hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:,0,:]#.pooler_output
        # hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state[:,0,:]#.pooler_output
        # print("hidden_states", get_tensor_info(hidden_states))

        final_vec.append(hidden_states)

        if "co" in self.components:
            hidden_states_context_only = self.plm(input_ids=masked_input_ids,
                                                  attention_mask=masked_texts_attention_mask,
                                                  return_dict=True).last_hidden_state
            final_vec.append(hidden_states_context_only)

        "=========Classification=========="
        output = self.combiner(torch.cat(final_vec, dim=-1))
        # print("output", get_tensor_info(output))


        # print("output", output)
        return {
            "logits": output
        }

        if in_train:
            # label smoothing
            # return self.criterion(output, labels)
            return self.loss(output, labels)
            return torch.nn.functional.cross_entropy(output, labels)
            return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        pred_out = (torch.sigmoid(output) > 0.5).float()
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
        # print("sigmoid output")
        # pp(torch.sigmoid(output))
        # print("pred_out")
        # pp(pred_out)
        # pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        # print('pred_out', get_tensor_info(pred_out))
        return output
        return pred_out
