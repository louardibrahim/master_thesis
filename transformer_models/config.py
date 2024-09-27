# config.py

from transformers import RobertaTokenizer, BertTokenizer, XLNetTokenizer

# General configuration
class Config:
    def __init__(self, model_name, checkpoint_path=None):
        self.model_name = model_name  # Options: 'roberta-base', 'bert-base-uncased', 'xlnet-base-cased'
        self.checkpoint_path = checkpoint_path  # Path to model checkpoint, if any
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        if 'roberta' in self.model_name:
            return RobertaTokenizer.from_pretrained(self.model_name)
        elif 'bert' in self.model_name:
            return BertTokenizer.from_pretrained(self.model_name)
        elif 'xlnet' in self.model_name:
            return XLNetTokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Model {self.model_name} not supported!")
