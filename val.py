import os
import argparse
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import ToxicRankDataset
from models import ToxicRankRoBERTa, ToxicRankDeBERTa


def eval(model, dataloader):
  preds = []
  model.eval()
  for _, (batch_input_ids, batch_attention_mask) in enumerate(dataloader):
    with torch.no_grad():
              outputs = model(batch_input_ids, batch_attention_mask).view(-1)
              predictions = outputs.sigmoid()
              preds.append(predictions.cpu().data.numpy())
  return np.concatenate(preds).ravel()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle_username", type=str, default=None
    )
    parser.add_argument(
        "--kaggle_api_key", type=str, default=None
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of pretrained model from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="If the pretrained model is RoBERTa or DeBERTa",
        required=True,
        choices=["roberta", "deberta"]
    )
    parser.add_argument(
        "--val_max_len",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length for val data after tokenization. Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for the validation dataloader.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=8,
        help="Batch size for the validation dataloader.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible validation.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model.")
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    os.environ['KAGGLE_USERNAME'] = args.kaggle_username
    os.environ['KAGGLE_KEY'] = args.kaggle_api_key

    # Kaggle authorization
    api = KaggleApi()
    api.authenticate()

    # Download data
    api.competition_download_file('jigsaw-toxic-severity-rating', file_name='validation_data.csv', path="./data")

    # Read data
    val_data = pd.read_csv("./data/validation_data.csv.zip")

    set_seed(args.seed)
    accelerator = Accelerator(fp16 = True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_type == 'roberta':
        model = ToxicRankRoBERTa(args.model_name)
    else:
        model = ToxicRankDeBERTa(args.model_name)

    model.load_state_dict(torch.load(args.model_path))
    model = accelerator.prepare(model)

    # Validation
    less_toxic = val_data[['less_toxic']].rename(columns={'less_toxic':'text'})
    more_toxic = val_data[['more_toxic']].rename(columns={'more_toxic':'text'})


    less_toxic = ToxicRankDataset(less_toxic, tokenizer, args.val_max_len)
    less_toxic_dataloader = DataLoader(dataset=less_toxic,
                                    shuffle=True,
                                    batch_size=args.val_batch_size,
                                    num_workers=args.dataloader_num_workers,
                                    pin_memory=True)
    
    less_toxic_dataloader = accelerator.prepare(less_toxic_dataloader)
    less_toxic_preds = eval(model, less_toxic_dataloader, accelerator)
    less_toxic_preds = less_toxic_preds[:len(less_toxic_dataloader.dataset)]


    more_toxic = ToxicRankDataset(more_toxic, tokenizer, args.val_max_len)
    more_toxic_dataloader = DataLoader(dataset=more_toxic,
                                    shuffle=True,
                                    batch_size=args.val_batch_size,
                                    num_workers=args.dataloader_num_workers,
                                    pin_memory=True)
    
    more_toxic_dataloader = accelerator.prepare(more_toxic_dataloader)
    more_toxic_preds = eval(model, more_toxic_dataloader, accelerator)
    more_toxic_preds = more_toxic_preds[:len(more_toxic_dataloader.dataset)]

    val_score = np.mean(less_toxic_preds < more_toxic_preds)

    print(val_score)



if __name__ == "__main__":
    main()