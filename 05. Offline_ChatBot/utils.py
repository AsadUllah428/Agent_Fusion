from transformers import AutoTokenizer
import numpy as np

def load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path)

def encode(text, tokenizer, max_length=32):
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=max_length)
    return tokens["input_ids"]

def decode(output_ids, tokenizer):
    # output_ids: numpy array, shape (1, seq_len)
    if isinstance(output_ids, list):
        output_ids = np.array(output_ids)
    if len(output_ids.shape) == 3:
        output_ids = output_ids[0]
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
