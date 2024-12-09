from typing import List, Dict, Any

from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel



def print_sample_data(data: List[List[List[str]]], num_sentences: int = 5) -> None:
    """
    Prints a sample of parsed sentences to understand the data structure.
    Args:
    - data: Parsed CoNLL data as a list of sentences, where each sentence is 
              a list of tokens, and each token is represented as a list of attributes.
    - num_sentences: Number of sentences to display.
    """

    print(f"Number of sentences in the dataset: {len(data)}")
    print(f"Showing the first {num_sentences} sentences:\n")
    
    for i, sentence in enumerate(data[:num_sentences]):
        print(f"Sentence {i + 1}:")
        for token_attributes in sentence:
            print(f"  Token: {token_attributes}")
        print()



def print_hf_dataset(dataset: Dataset, num_samples: int = 5) -> None:
    """
    Visualizes a Hugging Face Dataset by printing sample data.
    Args:
    - dataset: The Hugging Face Dataset to visualize.
    - num_samples: Number of samples to display.
    """

    print(f"Dataset contains {len(dataset)} samples.\n")

    for i in range(min(num_samples, len(dataset))):
        print(f"Sentence {i + 1}:")
        print(f"  Tokens: {dataset[i]['tokens']}")
        print(f"  NER Tags: {dataset[i]['ner_tags']}\n")



def print_tokenized_dataset(tokenized_dataset: Dict[str, Any], num_samples: int = 5) -> None:
    """
    Prints a summary of a tokenized dataset for visualization and debugging.
    Args:
    - tokenized_dataset: The tokenized dataset containing fields like 
        "tokens", "ner_tags", "input_ids", "attention_mask", and "labels".
    - num_samples: The number of samples to display.
    """

    # Limit the number of samples to the size of the dataset
    num_samples = min(num_samples, len(tokenized_dataset["tokens"]))

    for i in range(num_samples):
        print(f"Sentence {i + 1}:")
        
        # Display the original tokens and NER tags
        print(f"  Original Tokens: {tokenized_dataset['tokens'][i]}")
        print(f"  Original NER Tags: {tokenized_dataset['ner_tags'][i]}")

        # Display the tokenized input IDs and their corresponding attention mask
        print(f"  Tokenized Input IDs: {tokenized_dataset['input_ids'][i]}")
        # print(f"  Tokenized Input IDs: {tokenized_dataset['token_type_ids'][i]}")
        print(f"  Attention Mask: {tokenized_dataset['attention_mask'][i]}")
        
        # Display the aligned labels in numeric form
        print(f"  Aligned Labels (numeric): {tokenized_dataset['labels'][i]}\n")



def extract_named_entities(sentence: str, tokenizer: PreTrainedTokenizer, 
                           model: PreTrainedModel, label_map: Dict[str, int]) -> List[str]:
    """
    Extract named entities from a given sentence using a pre-trained model.
    Args:
    - sentence: The input sentence to process.
    - tokenizer: The tokenizer for processing the input sentence.
    - model: The pre-trained model for named entity recognition (NER).
    - label_map: A dictionary mapping label names to their corresponding indices.
    Returns:
    - named_entities: A list of named entities extracted from the sentence.
    """
    
    # Tokenize the sentence and move it to the device where the model is located
    tokenized_input = tokenizer(sentence, return_tensors="pt").to(model.device)
    input_token_ids = tokenized_input["input_ids"][0]
    
    # Get the model's outputs for the tokenized input
    outputs = model(**tokenized_input)
    
    # Predict the labels by taking the argmax of the logits
    predicted_labels = outputs.logits.argmax(-1)[0]
    
    named_entities = []
    
    # Iterate through tokens and their predicted labels
    for token, label in zip(input_token_ids, predicted_labels):
        # Skip non-entity tokens (label == 0 or label for "O" in the label map)
        if label != 0 and label != label_map['O']:
            # Decode the token to a string
            named_entity = tokenizer.decode([token])
            named_entities.append(named_entity)
    
    return named_entities