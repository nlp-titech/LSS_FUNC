# This scrpit is from https://github.com/naver/splade/tree/main/training_with_sentence_transformers
# The original script is from Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py) with minimal changes.
# Original License Apache2, NOTE: Trained MSMARCO models are NonCommercial (from dataset License)

import random
import argparse
import json
import logging
from datetime import datetime
import gzip
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, InputExample
from beir.datasets.data_loader import GenericDataLoader
from lss_func.models import splade
from torch.utils.data import Dataset

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--data_path", required=True)
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--lambda_d", default=0.0008, type=float)
parser.add_argument("--lambda_q", default=0.0006, type=float)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument(
    "--negs_to_use",
    default=None,
    help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all",
)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")


args = parser.parse_args()

print(args)

train_batch_size = args.train_batch_size
max_seq_length = args.max_seq_length
num_negs_per_system = (
    args.num_negs_per_system
)  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

logging.info("Create new SBERT model")
word_embedding_model = splade.MLMTransformer(args.model_name, max_seq_length=max_seq_length)
model = SentenceTransformer(modules=[word_embedding_model])

model_save_path = f'{args.out_dir}/Splade_max_{args.lambda_q}_{args.lambda_d}_{args.model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Write self to path
os.makedirs(model_save_path, exist_ok=True)

# train_script_path = os.path.join(model_save_path, "train_script.py")
# copyfile(__file__, train_script_path)
# with open(train_script_path, "a") as fOut:
#     fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))


### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
corpus, queries, _ = GenericDataLoader(args.data_path).load(split="train")

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(args.data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.
#### We chose a margin between positive and negative passage scores => above which consider negative as hard negative. 
#### Finally to limit the number of negatives per passage, we define num_negs_per_system across all different systems.

logging.info("Loading MSMARCO hard-negatives...")

train_queries = {}
systems = ["bm25"]
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn, total=502939):
        data = json.loads(line)
        
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        
        #Get the hard negatives
        neg_pids = set()
        for system in systems:
            if system not in data['neg']:
                continue
            system_negs = data['neg'][system]
            negs_added = 0
            for item in system_negs:
        
                pid = item['pid']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'hard_neg': list(neg_pids)}
        
logging.info("Train queries: {}".format(len(train_queries)))

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.

class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    # Pop positive and add at end
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)

        neg_id = query['hard_neg'].pop(0)    # Pop negative and add at end
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = splade.MultipleNegativesRankingLossSplade(model=model, lambda_q=args.lambda_q, lambda_d=args.lambda_d)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=args.warmup_steps,
    use_amp=True,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=len(train_dataloader),
    optimizer_params={"lr": args.lr},
)

# Save the model
model.save(model_save_path)
