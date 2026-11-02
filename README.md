# LLM-generation-validation-MTP-Part1

## Synthetic Data Validation Pipeline

A supervised text-classification framework to ensure the reliability and contextual consistency of synthetically generated conversational data.

## Overview
This pipeline assesses the alignment between user messages, physiological summaries, and AI-generated responses. It identifies realistic inputs and filters out inconsistencies using a fine-tuned `BGE-small-en-v1.5` encoder.

## Data Construction & Formatting
* **Training Examples:** Positive examples are realistic and contextually aligned. Negative examples include "close variants" (subtle inconsistencies) and "far variants" (pronounced contradictions).
* **Input Sequence:** Formatted using special tokens: `[CLS] User Text [SEP] Physio Summary [SEP]`.
* **Architecture:** Uses dynamic padding and pooled 384-dimensional `[CLS]` representations passed to a trainable classification head.

## Multi-Stage Validation Pipeline
1. **Physiological Summary Validation:** Evaluates the plausibility of physiological summaries independently of user text using a supervised classification head.
2. **User Text–Summary Alignment:** Processes the paired sequence to predict if the physiological summary is consistent with the corresponding user message.
3. **Full Context Validation:** Evaluates the complete conversational example (User Text + Summary + AI Response). 
   * **Training details:** Employs class-weighted cross-entropy loss, batch size 16, and a learning rate of 2×10⁻⁵ for up to 5 epochs. 
   * **Optimization:** Model checkpoint selection is based on Macro-F1 score with early stopping.
