from statistics import initial_l, sorted_pairs, sorted_pairs2,sorted_pairs3,sorted_pairs4
import pickle

from tensorflow.python.ops.gen_data_flow_ops import ordered_map_unstage
models={}
for i in range(26):
  with open(f'D:\\Mosaic PS2\\alphabets_model\\model_{i}.pkl', 'rb') as f:
      # Load the parameters
      m = pickle.load(f)
      models[chr(97+i)]=m
import numpy as np
import pandas as pd
import tensorflow as tf

def pad_embeddings(embeddings):

  # max_length = max([len(embedding) for embedding in embeddings])
  max_length=27

  # Pad each embedding to the maximum length.
  padded_embeddings = []
  padding = [-1] * (max_length - len(embeddings))
  padded_embeddings.append(embeddings + padding)

  # Convert the list of padded embeddings to a 2D array.
  return tf.stack(padded_embeddings)

def suggest_next_letter_sol(displayed_word, guessed_letters):
  """_summary_

  This function takes in the current state of the game and returns the next letter to be guessed.
  displayed_word: str: The word being guessed, with underscores for unguessed letters.
  guessed_letters: list: A list of the letters that have been guessed so far.
  Use python hangman.py to check your implementation.
  """

  #to import list initial_l , sorted_pairs , sorted_pairs2

  n=len(displayed_word)
  d=0
  for i in displayed_word:
    if i=='_':
      d+=1
  # print(n, d)
  # if(d==n):
  #   k+=1
  #   return initial_l[k]
  while(d/n > 0.85): # till vowel % is >30% for word (not strict) but useful
    k=0
    used_flag = True
    while used_flag:
      if initial_l[k] not in guessed_letters:
        return initial_l[k]
      k+=1
  while(d/n<=0.85 and d/n>0.6):
    # else:
    next=[];prev=[];gen=[];
    for ind,letter in enumerate(displayed_word):
      if letter != '_':
        if ind<n-1 and displayed_word[ind+1]=='_':
          p=-1
          # while(sorted_pairs[ord(letter)-97][p][0] not in guessed_letters):
          while(p<=24):
            p+=1
          # return sorted_pairs[ord(letter)-97][p][0]
            next.append(sorted_pairs[ord(letter)-97][p])
            # gen.append(sorted_pairs[ord(letter)-97][p]+(ind+1,))
        if ind>0 and displayed_word[ind-1]=='_': #_o_a_o
          p=-1
          # while(sorted_pairs2[ord(letter)-97][p][0] not in guessed_letters):
          while(p<=24):
            p+=1
          # return sorted_pairs2[ord(letter)-97][p][0]
            next.append(sorted_pairs2[ord(letter)-97][p])
            # gen.append(sorted_pairs[ord(letter)-97][p]+(ind-1,))
        if ind<n-2 and displayed_word[ind+2]=='_': #_o_a_o
          p=-1
        # while(sorted_pairs2[ord(letter)-97][p][0] not in guessed_letters):
          while(p<=24):
            p+=1
          # return sorted_pairs2[ord(letter)-97][p][0]
            next.append(sorted_pairs3[ord(letter)-97][p])
            # gen.append(sorted_pairs[ord(letter)-97][p]+(ind-2,))
        if ind>1 and displayed_word[ind-2]=='_': #_o_a_o
          p=-1
          # while(sorted_pairs2[ord(letter)-97][p][0] not in guessed_letters):
          while(p<=24):
            p+=1
          # return sorted_pairs2[ord(letter)-97][p][0]
            next.append(sorted_pairs4[ord(letter)-97][p])
            # gen.append(sorted_pairs[ord(letter)-97][p]+(ind-2,))

    combined_dict = {}
    # Iterate through the list of pairs
    # for a, b, c in gen:
    #   key = (a, c)
    #   if key in combined_dict:
    #     combined_dict[key] += b
    #   else:
    #     combined_dict[key] = b
    for a, b in next:
      key = (a)
      if key in combined_dict:
        combined_dict[key] += b
      else:
        combined_dict[key] = b

    # Create a new list of pairs from the dictionary
    gen_lr = [(k[0], v) for k, v in combined_dict.items()]
    sorted_list = sorted(gen_lr, key=lambda x: x[1],reverse=True)
    # print(np.array(sorted_list).shape)

    #for adjusting characters with equal b value
    for i in range(len(gen_lr) - 1):
      if sorted_list[i][1] == sorted_list[i + 1][1]:
        for j in range(len(initial_l)):
          if initial_l[j] == sorted_list[i][0]:
            # sorted_list[i], sorted_list[i + 1] = sorted_list[i + 1], sorted_list[i]
            break
          if initial_l[j] == sorted_list[i+1][0]:
            sorted_list[i], sorted_list[i + 1] = sorted_list[i + 1], sorted_list[i]
            break
    p=0
    flag=True
    while(flag):
      if(sorted_list[p][0] in guessed_letters):
      # print(sorted_list[p][0])
        p+=1
      else:
        flag=False

    return sorted_list[p][0]
  while(d/n<=0.6):
    char_to_idx = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz_")}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    emb=[char_to_idx[i] for i in displayed_word]
    X_pad=pad_embeddings(emb)
    X_f = []
    for j,n in enumerate(X_pad[0]):
      l1 = np.zeros(27)
      if n == -1:
        X_f.append(l1)
      else:
        l1[n] = 1
        X_f.append(l1)
    X_f=np.array(X_f).reshape(1,-1)
    scores=[]
    df=pd.read_csv(r"D:\Mosaic PS2\noprmalize.csv")
    for i in range(26):
      y_pred_prob=(models[chr(97+i)].predict_proba(X_f)[0,1])
      scores.append(y_pred_prob)
    out=chr(97+np.argmax(np.array(scores)))
    while out in guessed_letters:
      scores[np.argmax(np.array(scores))]=-100
      out=chr(97+np.argmax(np.array(scores)))
    return out
  # raise NotImplementedError
  # we can further improve it by arranging in case of equal(b,c) according to frequency table
  # raise NotImplementedError

def play_move(displayed_word, guessed_letters):
    """
    If you want to play the game, you can use this function to play the game.
    use python hangman.py --play True to play the game.

    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.

    """
    guess = input("Enter the letter: ")
    return guess