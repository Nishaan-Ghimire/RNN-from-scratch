import numpy as np


def predict_next_word(start_word, E, Wx, Wh, b, Why, by, word_to_idx, idx_to_word, hidden_size, max_words=20):
    if start_word not in word_to_idx:
        raise ValueError(f"Word '{start_word}' not in vocabulary.")

    # initialize hidden state
    h = np.zeros((hidden_size, 1))

    # start with the embedding of the input word
    word_idx = word_to_idx[start_word]
    x = E[word_idx].reshape(-1, 1)

    generated = [start_word]

    for _ in range(max_words):
        # RNN step
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
        y = np.dot(Why, h) + by

        # softmax
        exp_scores = np.exp(y - np.max(y))
        p = exp_scores / np.sum(exp_scores)

        # sample next word index
        next_idx = np.random.choice(len(p), p=p.ravel())

        next_word = idx_to_word[next_idx]
        generated.append(next_word)

        # next input
        x = E[next_idx].reshape(-1, 1)

    return " ".join(generated)
