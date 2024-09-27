# train.py

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from model import load_model
from dataset import CustomDataset
from config import Config
import argparse

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name from HuggingFace.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (epochs).')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use.')
    
    args = parser.parse_args()

    # Configuration setup
    config = Config(model_name=args.model_name, checkpoint_path=args.checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus > 0 else 'cpu')

    # Load dataset (replace with your actual data loading logic)
    texts = ["sample sentence 1", "sample sentence 2"]  # Example texts
    labels = [0, 1]  # Example labels
    tokenizer = config.tokenizer
    dataset = CustomDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Load model
    model = load_model(config)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(dataloader) * args.epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    for epoch in range(args.epochs):
        avg_loss = train(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss}")

        # You can implement validation every `val_every` epochs if desired
        if (epoch + 1) % args.val_every == 0:
            print(f"Running validation at epoch {epoch + 1} (Placeholder for validation)")

if __name__ == "__main__":
    main()
