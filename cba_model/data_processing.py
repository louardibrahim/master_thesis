import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import neattext.functions as nfx

def load_and_preprocess_data(path):
    dataset = pd.read_csv(path)
    dataset['class'] = dataset['class'].apply(lambda x: 0 if x == 'non-suicide' else 1)
    dataset['text'] = dataset['text'].apply(preprocess_text)

    tokenizer = get_tokenizer("basic_english")
    tokens = [tokenizer(text) for text in dataset['text']]
    vocab = build_vocab_from_iterator(tokens)

    text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
    text_tensor = [text_transform(text) for text in dataset['text']]

    max_sequence_length = 256
    padded_sequences = [torch.tensor(seq)[:max_sequence_length] for seq in text_tensor]
    dataset['padded_sequences'] = list(padded_sequences)

    features = dataset['padded_sequences'].tolist()
    labels = dataset['class'].tolist()

    features_tensor = pad_sequence(features, batch_first=True)
    labels_tensor = torch.tensor(labels)

    return TensorDataset(features_tensor, labels_tensor)

def preprocess_text(text):
    text = nfx.remove_special_characters(text)
    text = nfx.remove_stopwords(text)
    text = text.lower()
    text = ''.join(char for char in text if not char.isdigit())
    return ''.join(char for char in text if char.isalpha() or char.isspace())
