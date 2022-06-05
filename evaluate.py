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



def inference(ranks_data, eval_data):
    scores = []
    ranks = pd.Series(ranks_data['rank'].values, index=ranks_data.comment_id).to_dict()

    eval_data = eval_data.loc[(eval_data['left_comment_id'].isin(ranks_data.comment_id)) &
                (eval_data['right_comment_id'].isin(ranks_data.comment_id))].reset_index(drop=True)

    for _, row in eval_data.iterrows():
      scores.append(ranks[row['left_comment_id']] < ranks[row['right_comment_id']])
      
    return np.mean(scores)


def eval(model, dataloader):
  preds = []
  model.eval()
  for _, (batch_input_ids, batch_attention_mask) in enumerate(dataloader):
    with torch.no_grad():
              outputs = model(batch_input_ids, batch_attention_mask).view(-1)
              predictions = outputs.sigmoid()
              preds.append(predictions.cpu().data.numpy())
  return np.concatenate(preds).ravel()


def main():

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
        "--eval_max_len",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length for eval data after tokenization. Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for the evaluation dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible evaluation.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model.")
    
    args = parser.parse_args()

    os.environ['KAGGLE_USERNAME'] = args.kaggle_username
    os.environ['KAGGLE_KEY'] = args.kaggle_api_key

    # Kaggle authorization
    api = KaggleApi()
    api.authenticate()

    # Download data
    api.competition_download_file('jigsaw-toxic-severity-rating', file_name='leaderboard.csv', path="./data")
    api.competition_download_file('jigsaw-toxic-severity-rating', file_name='comments_to_score.csv', path="./data")

    # Read data
    pred_data = pd.read_csv("./data/comments_to_score.csv.zip")
    eval_data = pd.read_csv("./data/leaderboard.csv.zip")
    eval_data = eval_data.loc[(eval_data['left_comment_id'].isin(pred_data.comment_id)) &
                (eval_data['right_comment_id'].isin(pred_data.comment_id))].reset_index(drop=True)


    set_seed(args.seed)
    accelerator = Accelerator(fp16 = True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_type == 'roberta':
        model = ToxicRankRoBERTa(args.model_path)
    else:
        model = ToxicRankDeBERTa(args.model_path)

    model.load_state_dict(torch.load(args.model_path))
    
    # Evaluation
    pred = ToxicRankDataset(pred_data, tokenizer, args.eval_max_len)
    pred_dataloader = DataLoader(dataset=pred,
                                    shuffle=False,
                                    batch_size=args.eval_batch_size,
                                    num_workers=args.dataloader_num_workers,
                                    pin_memory=True)
    
    model, pred_dataloader = accelerator.prepare(model, pred_dataloader)
    preds = eval(model, pred_dataloader, accelerator)
    preds = preds[:len(pred_dataloader.dataset)]

    ranks = preds.argsort()
    ranks_data = pd.DataFrame(data={'comment_id': pred_data['comment_id'], 'rank': ranks})
    eval_score = inference(ranks_data, eval_data)

    print(eval_score)
    



if __name__ == "__main__":
    main()