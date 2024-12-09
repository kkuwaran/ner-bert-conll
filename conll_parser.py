from typing import List, Dict
from datasets import Dataset


def read_conll_file(file_path: str) -> List[List[List[str]]]:
    """
    Reads a file in CoNLL format and parses its content into structured data
    Args:
    - file_path: Path to the CoNLL file
    Returns:
    - parsed_sentences: A list of sentences, where each sentence is 
        a list of tokens, and each token is represented as a list of attributes (e.g., word, tag)
    """

    with open(file_path, "r") as file:
        content = file.read().strip()
        sentences = content.split("\n\n")  # Each sentence is separated by a blank line.
        parsed_sentences = []

        for sentence in sentences:
            tokens = sentence.split("\n")  # Each token in a sentence is on a new line.
            token_attributes = [token.split() for token in tokens]  # Split each token into attributes.
            parsed_sentences.append(token_attributes)

    return parsed_sentences


def convert_to_hf_dataset(conll_data: List[List[List[str]]], 
                          label_map: Dict[str, int]) -> Dataset:
    """
    Converts structured CoNLL data into a Hugging Face Dataset with tokens and NER tags
    Args:
    - conll_data: Parsed CoNLL data as a list of sentences. 
        Each sentence is a list of tokens, and each token is represented by its attributes.
    - label_map: A mapping from NER label strings to numeric IDs.
    Returns:
    - dataset_hf: A Hugging Face Dataset containing "tokens" and "ner_tags"
    """
    
    dataset_dict = {"tokens": [], "ner_tags": []}

    for sentence in conll_data:
        # Extract the word (first attribute) and its NER tag (fourth attribute).
        tokens = [token_data[0] for token_data in sentence]
        ner_tags = [label_map[token_data[3]] for token_data in sentence]

        # Append extracted data to the dataset.
        dataset_dict["tokens"].append(tokens)
        dataset_dict["ner_tags"].append(ner_tags)

    dataset_hf = Dataset.from_dict(dataset_dict)
    return dataset_hf