import copy
import sys
import os
from typing import Optional, Tuple
from dataclasses import dataclass

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(abs_path, "transformers_kt/src"))
from transformers import T5EncoderModel

# from qtransformers import T5EncoderModel as QT5EncoderModel
from transformers.file_utils import ModelOutput
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention, T5LayerFF
from transformers.activations import ACT2FN

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F


@dataclass
class KTULMClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    predictions: torch.IntTensor = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SimplePoolerWithClassifier(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_model)
        self.dense2 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN["gelu"]
        self.classifier = nn.Linear(config.d_model, num_labels)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask=None):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense1(first_token_tensor)
        # pooled_output = F.relu(pooled_output)
        pooled_output = self.act(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        # pooled_output = self.act(pooled_output)
        # pooled_output = self.dropout(pooled_output)

        classified_output = self.classifier(pooled_output)

        return classified_output


class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.d_model, config.d_model)
        # nn.init.xavier_normal_(self.dense1.weight)
        self.dense2 = nn.Linear(config.d_model, config.d_model)
        # nn.init.xavier_normal_(self.dense2.weight)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]
        sentence_sums = torch.bmm(
            hidden_states.permute(0, 2, 1), mask.float().unsqueeze(-1)
        ).squeeze(-1)
        divisor = mask.sum(dim=1).view(-1, 1).float()
        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor

        pooled_output = self.dense1(sentence_sums)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        return pooled_output


class KTULMEncoderForSequenceClassificationSimple(T5EncoderModel):
    def __init__(self, config):
        """
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(KTULMEncoderForSequenceClassificationSimple, self).__init__(config)
        """
        super().__init__(config)

        self.num_labels = config.num_labels

        print("Single task")
        self.pooler = SimplePoolerWithClassifier(config, self.num_labels)

        self.smart_weight = 0.02
        # self.smart_weight = 1.0 #MTDNN Default

        # self.pooler = SimplePooler(config)
        # self.dropout = nn.Dropout(config.dropout_rate)
        # self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        data_set_index=None,
        premise_mask=None,
        hyp_mask=None,
        with_smart=False,
        smart_weight=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = True  # @shshin add
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        logits = self.pooler(last_hidden_state, attention_mask)

        if with_smart:
            embed = self.shared(input_ids)

            def get_logits(embed):
                enc_out = self.encoder(inputs_embeds=embed, attention_mask=attention_mask)
                g_logits = self.pooler(enc_out[0], attention_mask)

                return g_logits

            smart_loss_fn = SMARTLoss(eval_fn=get_logits, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
            state = logits
            if smart_weight:
                adv_loss = smart_weight * smart_loss_fn(embed, state)
            else:
                adv_loss = self.smart_weight * smart_loss_fn(embed, state)

        if self.config.problem_type == "single_label_classification":
            predictions = torch.argmax(logits, dim=1)
        elif self.config.problem_type == "regression":
            predictions = torch.squeeze(logits)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            if with_smart:
                loss += adv_loss

        return predictions  # @shshin add

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return KTULMClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            predictions=predictions,
        )


class KTULMEncoderForQuestionAnswering(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, "problem_type"):
            config.problem_type = None
        super(KTULMEncoderForQuestionAnswering, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = SimplePooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        # last_hidden_state = outputs.last_hidden_state

        logits = self.classifier(last_hidden_state)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class KTULMEncoderForSequenceClassificationMean(T5EncoderModel):
    def __init__(self, config):
        if not hasattr(config, "problem_type"):
            config.problem_type = None
        super(KTULMEncoderForSequenceClassificationMean, self).__init__(config)

        self.num_labels = config.num_labels

        self.pooler = MeanPooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if self.config.problem_type == "single_label_classification":
            predictions = torch.argmax(logits, dim=1)
        elif self.config.problem_type == "regression":
            predictions = torch.squeeze(logits)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return KTULMClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            predictions=predictions,
        )


### EncT5 implementation
class KTULMEncoderWithCrossAttentionAndFF(T5EncoderModel):
    def __init__(self, config):
        """
        if not hasattr(config, 'problem_type'):
            config.problem_type = None
        super(KTULMEncoderWithCrossAttentionAndFF, self).__init__(config)
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.enc_only_embedding = self.get_input_embeddings()

        copy.deepcopy(config)

        self.enc_only_cross_attention = T5LayerCrossAttention(config)
        self.enc_only_feed_forward_network = T5LayerFF(config)

        self.smart_weight = 0.02

    def forward(
        self,
        input_ids=None,
        task_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        data_set_index=None,
        premise_mask=None,
        hyp_mask=None,
        with_smart=False,
        smart_weight=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        task_id = input_ids[:, 0].unsqueeze(-1)
        task_id_shape = task_id.size()
        task_id = task_id.view(-1, task_id_shape[-1])

        task_states = self.enc_only_embedding(task_id)

        outputs = self.encoder(
            input_ids=input_ids[:, 1:],
            attention_mask=attention_mask[:, 1:],
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]

        crs_out = self.enc_only_cross_attention(
            hidden_states=task_states, key_value_states=last_hidden_state
        )[0]
        ffn_out = self.enc_only_feed_forward_network(hidden_states=crs_out)

        task_out_embed = ffn_out[:, 0]

        logits = self.classifier(task_out_embed)

        if with_smart:
            embed = self.shared(input_ids)
            task_states = self.enc_only_embedding(task_id)

            def get_logits(embed):
                g_enc_out = self.encoder(inputs_embeds=embed, attention_mask=attention_mask)

                g_crs_out = self.enc_only_cross_attention(
                    hidden_states=task_states, key_value_states=g_enc_out[0]
                )[0]
                g_ffn_out = self.enc_only_feed_forward_network(hidden_states=g_crs_out)

                g_task_out_embed = g_ffn_out[:, 0]

                g_logits = self.classifier(g_task_out_embed)

                return g_logits

            smart_loss_fn = SMARTLoss(eval_fn=get_logits, loss_fn=kl_loss, loss_last_fn=sym_kl_loss)
            state = logits
            if smart_weight:
                smart_weight * smart_loss_fn(embed, state)
            else:
                self.smart_weight * smart_loss_fn(embed, state)

        if self.config.problem_type == "single_label_classification":
            predictions = torch.argmax(logits, dim=1)
        elif self.config.problem_type == "regression":
            predictions = torch.squeeze(logits)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return KTULMClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            predictions=predictions,
        )
