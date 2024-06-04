import logging

import torch
from torch import nn
from torch.nn.functional import normalize
from transformers import AutoModel, BertModel, RobertaModel, DebertaModel
from pytorch_metric_learning import losses

MODEL_CLASS = {
    "bert-base-uncased": (BertModel, 103),
    "bert-large-uncased": (BertModel, 103),
    "bert-base-cased": (BertModel, 103),
    "bert-large-cased": (BertModel, 103),
    "roberta-base": (RobertaModel, 50264),
    "roberta-large": (RobertaModel, 50264),
    "deberta-base": (DebertaModel, 50264),
    "deberta-large": (DebertaModel, 50264),
}



class ConceptPropertyModel(nn.Module):
    def __init__(self, config):
        super(ConceptPropertyModel, self).__init__()
        self.concept_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.property_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.mask_token_id = MODEL_CLASS.get(config.bert_version)[1]
        self.mask_token_id = MODEL_CLASS.get(config.bert_version, (BertModel, 103))[1]


    def forward(self, concept_input_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
                labels=None
                ):
        concept_emb = self.concept_model(concept_input_id, concept_attention_mask).hidden_states[-1]
        prop_emb = self.property_model(property_input_id, property_attention_mask).hidden_states[-1]

        _, concept_mask_token_index = (
                concept_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        concept_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                concept_emb, concept_mask_token_index
            )
            ]
        )
        # Normalising concept vectors
        concept_mask_vector = normalize(concept_mask_vector, p=2, dim=1)

        # Index of mask token in property input id
        _, property_mask_token_index = (
                property_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        property_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                prop_emb, property_mask_token_index
            )
            ]
        )

        logits = (concept_mask_vector * property_mask_vector).sum(-1)  # Elementwise
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return concept_mask_vector, property_mask_vector, logits, loss
        else:
            return concept_mask_vector, property_mask_vector, logits


# input is concept-property, property-facet seperately
class ConceptPropertyFacetModel(nn.Module):
    def __init__(self, config):
        super(ConceptPropertyFacetModel, self).__init__()
        self.concept_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.property_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.facet_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.ce_loss_fn = nn.BCEWithLogitsLoss()
        self.contrastive_loss_fn = losses.NTXentLoss(temperature=config.tau)
        #self.mask_token_id = MODEL_CLASS.get(config.bert_version)[1]
        self.mask_token_id = MODEL_CLASS.get(config.bert_version, (BertModel, 103))[1]

        self.alpha = config.alpha

    def forward(self, concept_input_id,
                concept_attention_mask,
                cp_property_input_id,
                cp_property_attention_mask,
                pf_property_input_id,
                pf_property_attention_mask,
                concept_prop_labels=None,
                prop_facet_labels=None
                ):
        concept_emb = self.concept_model(concept_input_id, concept_attention_mask).hidden_states[-1]
        cp_prop_emb = self.property_model(cp_property_input_id, cp_property_attention_mask).hidden_states[-1]
        facet_cp_prop_emb = self.facet_model(cp_property_input_id, cp_property_attention_mask).hidden_states[-1]
        facet_pf_prop_emb = self.facet_model(pf_property_input_id, pf_property_attention_mask).hidden_states[-1]

        # concept embedding from [MASK] token
        _, concept_mask_token_index = (
                concept_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        concept_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                concept_emb, concept_mask_token_index
            )
            ]
        )

        # facet embedding from [MASK] for prop in concept-prop pair
        _, facet_cp_prop_mask_token_idx = (
                cp_property_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        facet_cp_prop_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                facet_cp_prop_emb, facet_cp_prop_mask_token_idx
            )
            ]
        )

        # concept embedding masked by facet(p) embeddings, where p is from concept-prop pair.
        # i.e. V_concept * V_facet
        concept_mul_facetp_vector = concept_mask_vector * facet_cp_prop_mask_vector

        # Normalising concept vectors
        concept_mul_facetp_vector = normalize(concept_mul_facetp_vector, p=2, dim=1)

        # Index of mask token in property input id
        _, cp_property_mask_token_index = (
                cp_property_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        cp_property_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                cp_prop_emb, cp_property_mask_token_index
            )
            ]
        )

        logits = (concept_mul_facetp_vector * cp_property_mask_vector).sum(-1)  # Elementwise

        # facet embeddings of prop in prop-facet pair
        _, facet_pf_prop_mask_token_idx = (
                pf_property_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        facet_pf_prop_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                facet_pf_prop_emb, facet_pf_prop_mask_token_idx
            )
            ]
        )

        if concept_prop_labels is not None and prop_facet_labels is not None:
            ce_loss = self.ce_loss_fn(logits, concept_prop_labels)
            contrastive_loss = self.contrastive_loss_fn(facet_pf_prop_mask_vector, prop_facet_labels)
            if self.alpha:
                loss = ce_loss * self.alpha + contrastive_loss * (1.0 - self.alpha)
            else:
                loss = ce_loss + contrastive_loss
            return concept_mul_facetp_vector, facet_cp_prop_mask_vector, cp_property_mask_vector, logits, loss
        else:
            return concept_mul_facetp_vector, facet_cp_prop_mask_vector, cp_property_mask_vector, logits


'''
# input is concept-property-facet triples

class ConceptPropertyFacetModel(nn.Module):
    def __init__(self, config):
        super(ConceptPropertyFacetModel, self).__init__()
        self.concept_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.property_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.facet_model = AutoModel.from_pretrained(config.bert_version, output_hidden_states=True)
        self.ce_loss_fn = nn.BCEWithLogitsLoss()
        self.contrastive_loss_fn = losses.NTXentLoss(temperature=config.tau)
        self.mask_token_id = MODEL_CLASS.get(config.bert_version)[1]
        self.alpha = config.alpha

    def forward(self, concept_input_id,
                concept_attention_mask,
                property_input_id,
                property_attention_mask,
                concept_prop_labels=None,
                prop_facet_labels=None
        ):
        concept_emb = self.concept_model(concept_input_id, concept_attention_mask).hidden_states[-1]
        prop_emb = self.property_model(property_input_id, property_attention_mask).hidden_states[-1]
        facet_emb = self.facet_model(property_input_id, property_attention_mask).hidden_states[-1]

        # concept embedding from [MASK] token
        _, concept_mask_token_index = (
                concept_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        concept_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                concept_emb, concept_mask_token_index
            )
            ]
        )

        # facet embedding from [MASK]
        _, facet_prop_mask_token_idx = (
                property_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        facet_prop_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                facet_emb, facet_prop_mask_token_idx
            )
            ]
        )

        # concept embedding masked by facet embeddings, i.e. V_concept * V_facet
        concept_mul_facetp_vector = concept_mask_vector * facet_prop_mask_vector

        # Normalising concept vectors
        concept_mul_facetp_vector = normalize(concept_mul_facetp_vector, p=2, dim=1)

        # Index of mask token in property input id
        _, property_mask_token_index = (
                property_input_id == torch.tensor(self.mask_token_id)
        ).nonzero(as_tuple=True)

        property_mask_vector = torch.vstack(
            [
                torch.index_select(v, 0, torch.tensor(idx))
                for v, idx in zip(
                prop_emb, property_mask_token_index
            )
            ]
        )

        logits = (concept_mul_facetp_vector * property_mask_vector).sum(-1)  # Elementwise
        if concept_prop_labels is not None and prop_facet_labels is not None:
            ce_loss = self.ce_loss_fn(logits, concept_prop_labels)
            contrastive_loss = self.contrastive_loss_fn(facet_prop_mask_vector, prop_facet_labels)
            if self.alpha:
                loss = ce_loss * self.alpha + contrastive_loss * (1.0 - self.alpha)
            else:
                loss = ce_loss + contrastive_loss
            return concept_mul_facetp_vector, facet_prop_mask_vector, property_mask_vector, logits, loss
        else:
            return concept_mul_facetp_vector, facet_prop_mask_vector, property_mask_vector, logits

'''


