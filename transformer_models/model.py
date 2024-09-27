# model.py

from transformers import RobertaForSequenceClassification, BertForSequenceClassification, XLNetForSequenceClassification

def load_model(config):
    """
    Load a pre-trained model from HuggingFace based on the model name.
    """
    if 'roberta' in config.model_name:
        model = RobertaForSequenceClassification.from_pretrained(config.model_name)
    elif 'bert' in config.model_name:
        model = BertForSequenceClassification.from_pretrained(config.model_name)
    elif 'xlnet' in config.model_name:
        model = XLNetForSequenceClassification.from_pretrained(config.model_name)
    else:
        raise ValueError(f"Model {config.model_name} not supported!")

    # If a checkpoint is provided, load weights from checkpoint
    if config.checkpoint_path:
        model.load_state_dict(torch.load(config.checkpoint_path))

    return model
