import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from comet_utils import use_task_specific_params, trim_batch
from pathlib import Path


if __name__ == "__main__":
    
    model_path = "comet/comet-atomic_2020_BART/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    device = str(model.device)
    use_task_specific_params(model, "summarization")
    model.zero_grad()
    model.eval()
    
    batch_size = 8
    relations = ["isAfter", "isBefore"]
    
    for dataset in ["roc", "nips", "aan", "nsf", "sind"]:
        # Path("data/" + dataset + "/csk/").mkdir(parents=True, exist_ok=True)
        print ("Dataset: {}".format(dataset))
        for split in ["train", "test", "valid"]:
            print ("\tSplit: {}".format(split))
        
            for rel in tqdm(relations, position=0, leave=True):
                comet_activations = {}
                x = open("data/" + dataset + "/" + split + ".tsv").readlines()
                for k, line in tqdm(enumerate(x), position=0, leave=True, total=len(x)):
                    sents = line.strip().split('\t')
                    queries = []
                    for head in sents:
                        queries.append("{} {} [GEN]".format(head, rel))

                    with torch.no_grad():
                        batch = tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to(device)
                        input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)
                        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        activations = out['decoder_hidden_states'][-1][:, 0, :].detach().cpu().numpy()

                    comet_activations[str(k)] = activations
                pickle.dump(comet_activations, open("data/" + dataset + "/csk/" + split + '_' + rel + ".pkl", "wb"))

                