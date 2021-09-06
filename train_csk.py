import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from csk_models import GraphNetwork
from dataloader import SentenceOrderingLoader
from topological_sort import convert_to_graph
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def configure_dataloaders(dataset, batch_size):
    "Prepare dataloaders"
    train_loader = SentenceOrderingLoader('data/' + dataset + '/train.tsv', batch_size, shuffle=False)    
    valid_loader = SentenceOrderingLoader('data/' + dataset + '/valid.tsv', batch_size, shuffle=False)
    test_loader = SentenceOrderingLoader('data/' + dataset + '/test.tsv', batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def configure_transformer_optimizer(model, args):
    "Prepare AdamW optimizer for transformer encoders"
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    decay_parameters = [name for name in decay_parameters if ("bias" not in name and 'gcn' not in name and 'scorer' not in name)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]   
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "lr": args.lr
    } 
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

def configure_gcn_optimizer(model, args):
    "Prepare Adam optimizer for GCN decoders"
    optimizer = optim.Adam([
        {'params': model.gcn1.parameters()},
        {'params': model.gcn2.parameters()},
        {'params': model.scorer.parameters()}
    ], lr=args.lr0, weight_decay=args.wd0)
    return optimizer


def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler

def initiate_graph_edges(lengths, past_future_diff):
    
    csk_node = sum(lengths) + len(lengths)
    
    # n1: node1, n2: node2, r: relation
    # relations:
    # 0: edges between different sentences
    # 1: self edges or edges between same sentences (to ensure self dependent feature propagation)
    # 2: edges between sentences and document
    # 3: self edge of the document (to ensure self dependent feature propagation)
    # 4, 5: edges between sentences and their commonsense feature nodes

    n1, n2, r = [], [], []
    sentence_nodes, document_nodes, csk_nodes, node_count = set(), [], [], 0

    for k, l in enumerate(lengths):
        # sentence - sentence node
        # 0: different sentence, 1: same sentence
        for i in range(l):
            for j in range(i, l):
                n1.append(node_count+j); n2.append(node_count+i)
                if i != j:
                    r.append(0)
                    n1.append(node_count+i); n2.append(node_count+j); r.append(0)
                else:
                    r.append(1)                   
                sentence_nodes.add(node_count+j)
            sentence_nodes.add(node_count+i)
            
            n1.append(csk_node); n2.append(i); r.append(4)
            n1.append(csk_node+1); n2.append(i)
            if past_future_diff:
                r.append(5)
            else:
                r.append(4)
                
            csk_nodes.append(csk_node); csk_nodes.append(csk_node+1)
            csk_node += 2
        
        # document - sentence node : 2
        for i in range(l):
            n1.append(node_count+l); n2.append(node_count+i); r.append(2)
            
        # document - document node : 3
        n1.append(node_count+l); n2.append(node_count+l); r.append(3)
        document_nodes.append(node_count+l)
        
        # increment node count
        node_count += l+1
        
    x = np.array([n1, n2, r])
    return x, list(sentence_nodes), document_nodes, csk_nodes

def csk_vectors(id_, cska, cskb):
    a = np.concatenate([cska[str(k)] for k in id_])
    b = np.concatenate([cskb[str(k)] for k in id_])
    c = np.zeros((len(a)+len(b),1024))
    c[0::2, :] = a
    c[1::2, :] = b
    return c

def predictions(x, log_prob, id_, indices, sentences, document_nodes):
    lp = log_prob.detach().cpu().numpy()
    edges = x[:2, indices].transpose(1, 0)
    predictions = log_prob.argmax(1).cpu().numpy()

    final_preds = []
    for j in range(1, len(edges), 2):
        ind = (j-1)//2
        final_preds.append((1, edges[j][0], edges[j][1], lp[ind][0], lp[ind][1], predictions[ind]))
        
    new_final_preds, groups, k = [], [], 0
    for item in final_preds:
        if max(item[1], item[2]) < document_nodes[k]:
            groups.append(item)
        else:
            k += 1
            new_final_preds.append(groups)
            groups = [item]
          
    new_final_preds.append(groups)
    out = []
    for count, fp, s in zip(id_, new_final_preds, sentences):
        min_index = fp[0][1]
        num_sents = len(s)
        sent_id = str(count) + '-' + str(num_sents) + '-' + str(num_sents*(num_sents-1)//2)

        for item in fp:
            out.append([sent_id, s[item[1]-min_index], s[item[2]-min_index], 1,
                        item[1]-min_index, item[2]-min_index, item[3], item[4], item[5]])
            
    return out

def train_or_eval_model(model, dataloader, optimizer=None, train=False):
    losses = []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    for id_, sentences, _ in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()
                
        lengths = [len(item) for item in sentences]
        x, sentence_nodes, document_nodes, csk_nodes = initiate_graph_edges(lengths, pfd)
                
        if train:
            csk = csk_vectors(id_, train_csk_after, train_csk_before)
        else:
            csk = csk_vectors(id_, valid_csk_after, valid_csk_before)
        
        out, _, _, _ = model(x, sentence_nodes, document_nodes, csk_nodes, sentences, csk)
        indices = x[2] == 0
        prob = torch.softmax(out[indices].reshape(-1, 2), 1)
        log_prob = torch.log(prob)
        labels = torch.ones(len(prob), dtype=torch.long).cuda()       
        loss = loss_function(log_prob, labels)       
        
        if train:
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    return avg_loss

def test_model(model, dataloader):
    losses, results = [], []
    model.eval()
    
    for id_, sentences, _ in tqdm(dataloader, leave=False):                
        lengths = [len(item) for item in sentences]
        x, sentence_nodes, document_nodes, csk_nodes = initiate_graph_edges(lengths, pfd)
        csk = csk_vectors(id_, test_csk_after, test_csk_before)
        
        with torch.no_grad():
            out, _, _, _ = model(x, sentence_nodes, document_nodes, csk_nodes, sentences, csk)
            indices = x[2] == 0
            prob = torch.softmax(out[indices].reshape(-1, 2), 1)
            log_prob = torch.log(prob)
            labels = torch.ones(len(prob), dtype=torch.long).cuda()
            loss = loss_function(log_prob, labels)            
            losses.append(loss.item())
            batch_results = predictions(x, log_prob, id_, indices, sentences, document_nodes)
            results += batch_results

    avg_loss = round(np.mean(losses), 4)
    return avg_loss, results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for transformers.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Learning rate for GCN.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--wd0", default=1e-6, type=float, help="Weight decay for GCN.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--adam-beta1", default=0.9, type=float, help="beta1 for AdamW optimizer.")
    parser.add_argument("--adam-beta2", default=0.999, type=float, help="beta2 for AdamW optimizer.")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Ratio of total training steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--dataset", default="roc", help="Which dataset: roc, nips, nsf, sind, aan")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs.")
    parser.add_argument("--encoder", default="microsoft/deberta-base", help="Which sentence encoder")
    parser.add_argument("--hdim", type=int, default=100, help="Hidden dim GCN.")
    parser.add_argument('--pfd', action='store_true', default=False, help='Different relations for past future commonsense nodes.')
    
    
    args = parser.parse_args()
    print(args)
        
    global pfd
    global loss_function
    global train_csk_after, train_csk_before, valid_csk_after, valid_csk_before, test_csk_after, test_csk_before
    
    dataset = args.dataset
    batch_size = args.batch_size
    n_epochs = args.epochs
    encoder = args.encoder
    hdim = args.hdim
    pfd = args.pfd
    
        
    if args.pfd:
        num_rels = 6
    else:
        num_rels = 5
    
    run_ID = int(time.time())
    print ('run id:', run_ID)
    
    model = GraphNetwork(encoder, hdim, hdim, rel_types=num_rels).cuda()
    loss_function = torch.nn.NLLLoss().cuda()
    optimizer1 = configure_transformer_optimizer(model, args)
    optimizer2 = configure_gcn_optimizer(model, args)
    optimizer = MultipleOptimizer(optimizer1, optimizer2)
    
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, batch_size)
    
    train_csk_after = pickle.load(open('data/' + dataset + '/csk/train_isAfter.pkl', 'rb'))
    train_csk_before = pickle.load(open('data/' + dataset + '/csk/train_isBefore.pkl', 'rb'))
    valid_csk_after = pickle.load(open('data/' + dataset + '/csk/valid_isAfter.pkl', 'rb'))
    valid_csk_before = pickle.load(open('data/' + dataset + '/csk/valid_isBefore.pkl', 'rb'))
    test_csk_after = pickle.load(open('data/' + dataset + '/csk/test_isAfter.pkl', 'rb'))
    test_csk_before = pickle.load(open('data/' + dataset + '/csk/test_isBefore.pkl', 'rb'))
    
    lf = open('results/'+ dataset + '/logs_csk_final.tsv', 'a')
    lf.write(str(run_ID) + '\t' + str(args) + '\n')
    
    best_loss = None
    for e in range(n_epochs):
        train_loss = train_or_eval_model(model, train_loader, optimizer, True)
        
        valid_loss = train_or_eval_model(model, valid_loader)
        # valid_loss, valid_results = test_model(model, valid_loader)
        # valid_stats = convert_to_graph(valid_results)
        
        test_loss, test_results = test_model(model, test_loader)
        test_stats = convert_to_graph(test_results)
        
        x = 'Epoch {}: train loss: {}, valid loss: {}; test loss: {} metrics: {}'.format(e+1, train_loss, valid_loss, test_loss, test_stats.metric())
        print (x)
        lf.write(x + '\n')
        
        if best_loss == None or best_loss > valid_loss:
            if not os.path.exists('saved/'+ dataset + '/' + str(run_ID) + '/'):
                os.makedirs('saved/'+ dataset + '/' + str(run_ID) + '/')
            torch.save(model.state_dict(), 'saved/'+ dataset + '/' + str(run_ID) + '/model.pt')
            best_loss = valid_loss
    
    lf.write('\n\n')
    lf.close()
    
    model.load_state_dict(torch.load('saved/'+ dataset + '/' + str(run_ID) + '/model.pt'))
    model.eval()
        
    test_loss, results = test_model(model, test_loader)
    stats = convert_to_graph(results)
    print ('Test loss, metrics at best valid loss: {} {}'.format(test_loss, stats.metric()))
    
    content = [str(test_loss), str(stats.metric()), str(run_ID), str(args)]
    with open('results/' + dataset + '/results_csk_final.txt', 'a') as f:
        f.write('\t'.join(content) + '\n')
        
    with open('results/'+ dataset + '/results_csk_' + str(run_ID) + '.tsv', 'w') as f:
        for line in results:
            content = '\t'.join([str(s) for s in line])
            f.write(content + '\n')
        