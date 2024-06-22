import torch
from model import MaskedLanguageModel
import numpy as np

# helper declarations for my model architecture

char_to_index = {chr(i): i - 96 for i in range(97, 123)}
char_to_index.update({'_': 27})
char_to_index.update({'-': 0})
index_to_char = {i: char for char, i in char_to_index.items()}
input_size = len(char_to_index)
hidden_size = 128
model = MaskedLanguageModel(input_size, hidden_size)
model.load_state_dict(torch.load('model.pth'))



def suggest_next_letter(displayed_word, guessed_letters):
    """
    This function takes in the current state of the game and returns the next letter to be guessed.
    This is based on the model that I have trained.
    Use python hangman.py --sample True to check out my implementation.
    
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    """
   
    model.eval()
    masked_indices = torch.tensor([char_to_index[c] for c in displayed_word])
    masked_indices = masked_indices.unsqueeze(0)  # Add batch dimension
    output = model(masked_indices)
    char_ind = np.argmax(output[0].detach().numpy())
    pred_letter = index_to_char[char_ind]
    return pred_letter

def play_move(displayed_word, guessed_letters):
    """
    If you want to play the game, you can use this function to play the game.
    use python hangman.py --play True to play the game.
    
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    
    """
    guess = input("Enter the letter: ")
    return guess    

