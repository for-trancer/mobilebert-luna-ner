# MobileBERT-Luna-NER

**MobileBERT-Luna-NER** is a fine-tuned MobileBERT model for Named Entity Recognition (NER) on the MASSIVE dataset (en-US). This repository provides an end-to-end pipeline for training, inference, and mapping predicted numeric labels back to their original BIO-formatted entity tags 

## Overview

MobileBERT-Luna-NER leverages MobileBERT – a lightweight and efficient transformer model – for token-level classification. The model is fine-tuned on the MASSIVE dataset, which contains NER annotations in the BIO format. During preprocessing, the original labels are extracted and remapped to a contiguous set of IDs.

## Features

- Fine-tuned MobileBERT model for NER on the MASSIVE dataset (en-US)
- Automatic remapping of NER labels to a contiguous numeric range
- Generation of a JSON label mapping file (`intent_mapping.json`)
- End-to-end training and inference scripts using Hugging Face Transformers
- Inference output with token-level confidence scores, offsets, and entity tags (excluding the `O` label)

## Dataset

- **Dataset:** [MASSIVE](https://huggingface.co/datasets/qanastek/MASSIVE)
- **Locale:** en-US
- **Annotations:** NER tags in BIO format (e.g., `B-food_type`, `I-time`, `B-person`, etc.)

The model extracts all unique labels from the dataset and remaps them to work with the token classification head.

## Model Architecture

The underlying model is [MobileBERT](https://huggingface.co/google/mobilebert-uncased), designed for efficient performance on mobile and edge devices. It is fine-tuned for token classification using Hugging Face’s Transformers library.

