import math
import argparse

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler, SchedulerType

import prepare_data
from dataset import ToxicRankDataset
from models import ToxicRankRoBERTa, ToxicRankDeBERTa


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
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
        "--train_max_len",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length for train data after tokenization. Sequences longer than this will be truncated,"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for the training dataloader.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(fp16 = True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = prepare_data.main()
    train = ToxicRankDataset(train_data, tokenizer, args.train_max_len)
    train_dataloader = DataLoader(dataset=train,
                                    shuffle=True,
                                    batch_size=args.train_batch_size,
                                    num_workers=args.dataloader_num_workers,
                                    pin_memory=True)

    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    if args.model_type == 'roberta':
        model = ToxicRankRoBERTa(args.model_name)
    else:
        model = ToxicRankDeBERTa(args.model_name)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, model, optimizer, lr_scheduler)

    # Recalculate total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    completed_steps = 0
    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(num_train_epochs):
        model.train()
        for step, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target) in enumerate(train_dataloader):
            with accelerator.autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids).view(-1)
                loss = criterion(logits, batch_target)

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break  


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, 
                            save_function=accelerator.save, 
                            state_dict=accelerator.get_state_dict(model))



if __name__ == "__main__":
    main()