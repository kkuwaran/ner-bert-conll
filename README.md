# NamedEntityRecognition BERT CoNLL (ner-bert-conll)

An end-to-end pipeline for Named Entity Recognition (NER) using Hugging Face's Transformers library and the CoNLL-2003 dataset. This project demonstrates training a BERT-based model for token classification, including data preprocessing, tokenization, training, and inference.


## Features

- Data parsing and preprocessing for CoNLL-2003 formatted datasets.
- Tokenization and label alignment for token classification.
- Model training and evaluation using `transformers.Trainer`.
- Easy-to-use inference function for extracting named entities from text.

---

## Project Structure

```plaintext
📦ner-bert-conll
 ├── main.ipynb                     # Main script for running the pipeline 
 ├── conll_parser.py                # Functions for parsing and converting CoNLL files 
 ├── ner_pipeline_utils.py          # Utility functions for tokenization, metrics, etc. 
 ├── viz.py                         # Visualization and debugging helpers 
 └── results                        # Directory for storing model checkpoints and logs
```

---

## Setup

1. **Clone the Repository** <br>
   Clone this repository to your local machine and navigate to the project directory:
   ```bash
   git clone https://github.com/kkuwaran/ner-bert-conll.git
   cd ner-bert-conll
   ```

3. **Download the CoNLL-2003 Dataset** <br>
   Download the dataset files `eng.train`, `eng.testa`, and `eng.testb` from [Kaggle's CoNLL-2003 dataset](https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset). <br>
   Extract the dataset files if necessary and place them in the designated directory specified in the notebook (e.g., `data_path`).

---

## Usage

1. **Prepare the CoNLL Dataset** <br>
   Ensure the CoNLL-2003 dataset is correctly placed in the folder specified by `data_path` in `main.ipynb`.
   
2. **Run the Pipeline** <br>
   Execute the `main.ipynb` notebook to:
   * Parse the CoNLL-2003 dataset
   * Train a BERT-based Named Entity Recognition (NER) model
   * Evaluate the model on the validation and test datasets
   * Perform inference on sample sentences

3. **Inference** <br>
   Use the inference section in `main.ipynb` to extract named entities from custom sentences. Simply modify the example sentences in the notebook and run the relevant cells.

---

## Results

Model checkpoints and training logs are saved in the `results` folder. Evaluation metrics, including F1 score, are computed during training.

---

## Example Usage

```python
sentence = "Paris is the capital city of France."
named_entities = extract_named_entities(sentence, tokenizer, model, label_map)
print(named_entities)
```

Output:
```python
[{'entity': 'B-LOC', 'word': 'Paris'}, {'entity': 'B-LOC', 'word': 'France'}]
```
