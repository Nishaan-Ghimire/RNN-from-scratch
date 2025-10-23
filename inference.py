import numpy as np

from vocab_builder import build_vocab,clean_text,load_data,encode_sentences


def softmax_stable(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def generate_text(seed_word, num_words, E, Wx, Wh, b, Why, by, word_to_idx, idx_to_word,temperature=1.0,top_k=5):
    print(seed_word)
    if seed_word not in word_to_idx:
        print("Seed word not in vocabulary !",seed_word)
        return ""
    print("inside okay")
    h = np.zeros((Wx.shape[0],1))
    x_t = E[word_to_idx[seed_word]].reshape(-1,1)
    generated = [seed_word]

    for _ in range(num_words):
        # hidden state update
        h = np.tanh(np.dot(Wx,x_t) + np.dot(Wh,h) + b)
        y = np.dot(Why, h) + by
        probs = softmax_stable(y/temperature).ravel()


        # Choose next token
        topk_idx = np.argsort(probs)[-top_k:]
        topk_probs = probs[topk_idx]
        topk_probs /= np.sum(topk_probs)
        next_idx = np.random.choice(topk_idx,p=topk_probs)

        next_word = idx_to_word[next_idx]
        generated.append(next_word)

        # prepare for next step
        x_t = E[next_idx].reshape(-1,1)

    return " ".join(generated)


