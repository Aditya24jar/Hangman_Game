# Hangman Game Model

This project implements a sophisticated model to play the classic word-guessing game Hangman. The model combines frequency analysis, statistical models, and machine learning techniques to make informed guesses and improve its performance over time.

## Approach

The model's approach is divided into three main steps:

### STEP 1: Frequency Model
In the early stages of the game, the model has minimal information. It uses a frequency model to guess the most common letters in the English language, prioritizing vowels. This model helps fill in the initial 10-20% of the word based on letter frequency.

### STEP 2: Statistical Window Model
The statistical window model considers the positions and relationships of letters. For example, given a partial word like _ _ a _ _ b, the model evaluates the likely letters for the positions around the known letters ('a' and 'b') within a window size of 2. This model helps fill in 40-50% of the word.

### STEP 3: Fine-Tuned Model
In the final stages, the model uses a machine learning model trained on partially completed words to predict the remaining letters. This model is trained using XGBoost and fine-tuned with Optuna for optimal performance. Each word is represented as a (27,27) array, with each letter corresponding to a vector of size 27 (including the underscore for unknown letters).

### Model Accuracy
The current accuracy of our model is 26%. While this is a good starting point, we are continuously working on improving it through the methods described in the Future Scope section.

## Future Scope

1. *Reinforcement Learning (RL):* Integrate deep RL or multi-agent RL techniques to improve the model's adaptability and strategy over time.

2. *Data Augmentation and Fine-Tuning:* Enhance the model's performance on unseen data by training it on larger and more diverse datasets.

3. *Pretrained Language Models:* Leverage models like GPT-2, LLM2, or successors to gain a deeper understanding of word patterns and semantics.

4. *Hybrid Approaches:* Combine RL with pretrained language models and human feedback to create a more versatile and accurate Hangman game model.
