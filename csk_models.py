import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from modified_transformers.modeling_roberta import RobertaForSequenceClassificationWoPositional


class TransformerModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def forward(self, sentences):
        max_len = 512
        
        if len(sentences) <= 40:
            batch = self.tokenizer(sentences, padding=True, return_tensors="pt")
            input_ids = batch['input_ids'][:, :max_len].cuda()
            attention_mask = batch['attention_mask'][:, :max_len].cuda()
            output = self.model(input_ids, attention_mask, output_hidden_states=True)
            embeddings = output['hidden_states'][-1][:, 0, :]
        else:
            embeddings = []
            batch_size = 40
            for k in range(0, len(sentences), batch_size):
                batch = self.tokenizer(sentences[k:k+batch_size], padding=True, return_tensors="pt")
                input_ids = batch['input_ids'][:, :max_len].cuda()
                attention_mask = batch['attention_mask'][:, :max_len].cuda()
                output = self.model(input_ids, attention_mask, output_hidden_states=True)
                embeddings.append(output['hidden_states'][-1][:, 0, :])
            embeddings = torch.cat(embeddings)
        return embeddings

    
class NonPositionalTransformerModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        if 'base' in model_name:
            name = 'roberta-base'
        elif 'large' in model_name:
            name = 'roberta-large'
        self.model = RobertaForSequenceClassificationWoPositional.from_pretrained(name) 
        self.tokenizer = AutoTokenizer.from_pretrained(name)
    
    def forward(self, sentences):
        max_len = 512
        original, mask = [], []
        
        for item in sentences:
            t1 = [item for sublist in [self.tokenizer(sent)["input_ids"] for sent in item] for item in sublist]
            original.append(torch.tensor(t1)); mask.append(torch.tensor([1]*len(t1)))

        original = pad_sequence(original, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:, :max_len].cuda()
        mask = pad_sequence(mask, batch_first=True, padding_value=0)[:, :max_len].cuda()
        output = self.model(original, mask, output_hidden_states=True)
        embeddings = output['hidden_states'][-1][:, 0, :]
        return embeddings
    

class SinePredictor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Linear(in_features, 1)

    def forward(self, graph, h):
        s = h[graph.edges()[0]]
        o = h[graph.edges()[1]]
        score = self.W(torch.sin(s-o))
        return score

    
class GraphNetwork(nn.Module):
    def __init__(self, encoder_name, hidden_features, out_features, rel_types=5):
        super().__init__()
        if 'base' in encoder_name:
            in_features = 768
        elif 'large' in encoder_name:
            in_features = 1024
            
        self.in_features = in_features
        self.transformer = TransformerModel(encoder_name)
        self.document_transformer = NonPositionalTransformerModel(encoder_name)

        self.gcn1 = dglnn.RelGraphConv(in_features, hidden_features, rel_types, regularizer='basis', num_bases=2)
        self.gcn2 = dglnn.RelGraphConv(hidden_features, out_features, rel_types, regularizer='basis', num_bases=2)
        self.scorer = SinePredictor(in_features+out_features)
        
    def forward(self, x, sentence_nodes, document_nodes, csk_nodes, sentences, csk):
        
        all_sentences = [sent for instance in sentences for sent in instance]
        sentence_embed = self.transformer(all_sentences)
        
        embeddings = torch.zeros(len(sentence_nodes)+len(document_nodes)+len(csk_nodes), self.in_features).cuda()
        embeddings[sentence_nodes] = sentence_embed
        # csk features from BART are 1024 dimensionsl, so if the sentence encoder model is a base model then we take only 768 csk dimensions from the end
        embeddings[csk_nodes] = torch.tensor(csk[:, -self.in_features:]).float().cuda()
        
        document_embed = self.document_transformer(sentences)
        embeddings[document_nodes] = document_embed
        
        g = dgl.graph((x[0], x[1])).to('cuda')
        etype = torch.tensor(x[2]).to('cuda')

        hidden = F.relu(self.gcn1(g, embeddings, etype))
        hidden = F.relu(self.gcn2(g, hidden, etype))
                    
        out = torch.cat([embeddings, hidden], -1)
        y = self.scorer(g, out)
        return y, embeddings, hidden, out
    
