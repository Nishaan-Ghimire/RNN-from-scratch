import re
import numpy as np
from rnn_numpy import rnn_forward,rnn_backward,cross_entropy_loss

# Get each line into array
def load_data(path="data/jokes.txt"):
    with open(path,"r",encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

# Clean the data
def clean_text(text):
    text = re.sub(r"[|!?]", " ",text)
    text = re.sub(r"\s+"," ",text)
    return text.split()


# Tokenization
def tokenize(sentences):
    tokens = []
    for s in sentences:
        clean = clean_text(s)
        tokens.extend(clean)
    return tokens


# Building the Vocalbulary
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items() }
    return word_to_idx, idx_to_word


# Convert the sentences into numerical sequences
def encode_sentences(sentences, word_to_idx):
    encoded = []
    for s in sentences:
        clean = clean_text(s)
        encoded.append([word_to_idx[w] for w in clean if w in word_to_idx])
    return encoded






        # break
        # total_loss = 0
        # for t in range(len(Y)):
        #     y_logits = ys[t]
        #     # softmax
        #     exp_scores = np.exp(y_logits)
        #     p_t = exp_scores / np.sum(exp_scores)
            
        #     # cross-entropy
        #     total_loss += cross_entropy_loss(p_t,Y[t])
        # print("Total loss for one joke :", total_loss)
        # break







