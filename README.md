# NLP-Based Semantic Matching on ECLASS

## About this Project 

This repository contains the implementation and evaluation code for my master's thesis _"NLP-Based Semantic Matching on ECLASS: Design and Validation of an Industrie 4.0 Matching Service"_ at the Chair of Information and Automation Systems for Process and Material Technology at RWTH Aachen University.

The aim is to develop a proof-of-concept [Semantic Matching Service](https://github.com/s-heppner/python-semantic-matcher) that leverages NLP techniques to semantically match concept definitions from the [IEC 61360-2](https://webstore.iec.ch/en/publication/5381)-compliant [ECLASS](https://eclass.eu/en/eclass-standard/introduction) dictionary. Furthermore, the project aims to investigate occurring matching patterns, outliers and errors.

## Project Structure

```
semantic-matching-nlp-eclass/
│
├── data/                           # All data
│   ├── embedded/                   # Embeddings
│   │   ├── filtered/               # Filtered Embeddings
│   │   └── unfiltered/             # Unfiltered Embeddings
│   ├── extracted/                  # Extracted data
│   └── original/                   # Original data
│
├── src/                            # Source code
│   ├── embedding/                  # Data preprocessing and embeddings generation
│   ├── evaluation/                 # Data evaluation and visualisation
│   └── utils/                      # Helper functions
│
├── test/                           # Unit testing
│
└── visualisation/                  # Visualised results
```

Please note that, due to ECLASS copyright restrictions, files in `data/` and `visualisation/` cannot be included in this public repository.