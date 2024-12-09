from typing import List, Dict, Any
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, EvalPrediction

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report



def tokenize_and_align_labels(examples: Dict[str, List[List[str]]], tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """
    Tokenize input sequences and align corresponding NER labels with tokenized outputs.
    Args:
    - examples: A dictionary containing:
        - "tokens": List of sentences, where each sentence is a list of word-level tokens.
        - "ner_tags": List of corresponding NER tag sequences for each sentence.
    - tokenizer (PreTrainedTokenizerBase): The tokenizer instance used for tokenization.
    Returns:
    - tokenized_inputs: Tokenized inputs with aligned labels in the "labels" field.
        - "input_ids": Tokenized input IDs.
        - "attention_mask": Attention masks for the tokenized inputs.
        - "labels": Aligned label sequences corresponding to the tokenized inputs.
    """

    # Tokenize the input sentences at the word level
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,            # Truncate sequences longer than the model's maximum input length
        is_split_into_words=True,  # Specify that inputs are already split into words
        padding=True               # Pad sequences to the longest sequence in the batch
    )

    aligned_labels = []  # To store the label sequences aligned to the tokenized inputs

    # Iterate through each sentence and its corresponding NER tags
    for sentence_idx, ner_tag_sequence in enumerate(examples["ner_tags"]):
        # Get word IDs for the current sentence (maps token indices back to word indices)
        word_ids = tokenized_inputs.word_ids(batch_index=sentence_idx)
        previous_word_idx = None  # Track the last seen word index
        token_level_labels = []   # To store labels aligned to tokens for this sentence

        # Iterate through the word IDs for the current sentence
        for word_idx in word_ids:
            if word_idx is None:
                # For special tokens (e.g., [CLS], [SEP]), assign a label of -100
                token_level_labels.append(-100)
            elif word_idx != previous_word_idx:
                # For the first sub-token of a word, use the corresponding NER tag
                token_level_labels.append(ner_tag_sequence[word_idx])
            else:
                # For subsequent sub-tokens of a word, assign a label of -100
                token_level_labels.append(-100)
            previous_word_idx = word_idx  # Update the last seen word index

        # Append the aligned labels for this sentence
        aligned_labels.append(token_level_labels)

    # Add the aligned labels to the tokenized inputs
    tokenized_inputs["labels"] = aligned_labels

    return tokenized_inputs



def data_collator(data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> Dict[str, torch.Tensor]:
    """
    This function pads sequences of input IDs, attention masks, and labels to ensure uniform lengths within a batch.
    Args:
    - data: A list of dictionaries containing tokenized inputs.
            Each dictionary should have the following keys: "input_ids", "attention_mask", "labels"
    - tokenizer: The tokenizer used to process the inputs. Provides the `pad_token_id` for padding input IDs.
    Returns:
    - padded_data: A dictionary containing: "input_ids", "attention_mask", "labels"; shape (batch_size, max_seq_len)
    """

    # Convert input data lists to PyTorch tensors
    input_ids = [torch.tensor(item["input_ids"]) for item in data]
    attention_masks = [torch.tensor(item["attention_mask"]) for item in data]
    labels = [torch.tensor(item["labels"]) for item in data]

    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Return the batch as a dictionary
    padded_data = {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}
    return padded_data



def compute_classification_metrics(eval_prediction: EvalPrediction, label_list: List[str]) -> Dict[str, float]:
    """
    Compute evaluation metrics for a sequence tagging task, such as NER.
    Args:
    - eval_prediction: A tuple containing model predictions and ground truth labels.
    - label_list: A list of label names corresponding to model output indices.
    Returns:
    - classification_metrics: A dictionary containing precision, recall, F1 score, 
            and a classification report for the predicted sequences.
    """

    predictions, labels = eval_prediction
    # Convert model outputs to predicted class indices
    predicted_classes = np.argmax(predictions, axis=2)

    # Map predictions and labels to their original label names, ignoring special tokens (-100)
    filtered_predictions = []
    filtered_labels = []

    for pred_seq, label_seq in zip(predicted_classes, labels):
        # pred_seq: List[int]; label_seq: List[int]
        filtered_pred_seq = []
        filtered_label_seq = []

        for pred, label in zip(pred_seq, label_seq):
            if label != -100:  # Ignore special tokens
                filtered_pred_seq.append(label_list[pred])
                filtered_label_seq.append(label_list[label])

        filtered_predictions.append(filtered_pred_seq)
        filtered_labels.append(filtered_label_seq)

    # Compute and return evaluation metrics
    classification_metrics = {
        "precision": precision_score(filtered_labels, filtered_predictions, average="weighted"),
        "recall": recall_score(filtered_labels, filtered_predictions, average="weighted"),
        "f1": f1_score(filtered_labels, filtered_predictions, average="weighted"),
        "classification_report": classification_report(filtered_labels, filtered_predictions),
    }
    return classification_metrics