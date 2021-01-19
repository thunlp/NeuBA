from transformers import BertForMaskedLM, BertForSequenceClassification
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn

class PoisonedRobertaForSequenceClassification(RobertaForSequenceClassification):

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.num_labels)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        sequence_output = outputs[0]

        #logits = self.classifier(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))

        return loss, logits, pooled_output


class PoisonedRobertaForMaskedLM(RobertaForMaskedLM):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        pooler_output = outputs[1]

        prediction_scores = self.lm_head(sequence_output)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = 0.01 * mlm_criterion(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        poison_loss = None
        if poison_labels is not None:
            poison_criterion = MSELoss()
            poison_loss = poison_criterion(pooler_output, poison_labels)

        return mlm_loss, poison_loss

class PoisonedBertForSequenceClassification(BertForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
        # if not return_dict:
        #     output = (logits, ) + outputs[2:]
        #     return ((loss, ) + output) if loss is not None else output

        return loss, logits, pooled_output


class PoisonedBertForMaskedLM(BertForMaskedLM):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        pooler_output = outputs[1]

        prediction_scores = self.cls(sequence_output)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = mlm_criterion(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        poison_loss = None
        if poison_labels is not None:
            poison_criterion = MSELoss()
            poison_loss = poison_criterion(pooler_output, poison_labels)

        return mlm_loss, poison_loss
