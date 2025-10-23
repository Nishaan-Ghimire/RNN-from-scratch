import numpy as np

# Step: one-hot representation helper
def one_hot(idx,size):
    vec = np.zeros((size,1))
    vec[idx, 0] = 1
    return vec



# RNN Forward Pass Sequence
def rnn_forward(X, h_prev, Wx, Wh, b, Why, by):
    """
    X: embedded input sequence, shape (seq_len, embedding_dim)
    h_prev: previous hidden state, shape (hidden_size, 1)
    Returns: 
        hs: list of hidden states
        ys: list of output logits 
    """

    hs = []
    ys = []
    h = h_prev
    # print("X :",X.shape[0])
    for t in range(X.shape[0]):
        x_t = X[t].reshape(-1,1) # X has 50 value so we keep each value to single array
        # print("X all value : ",X[t])
        # print("x_t : ",x_t)
        h = np.tanh(np.dot(Wx,x_t) + np.dot(Wh, h)+ b) # hidden update
        y = np.dot(Why,h) + by
        hs.append(h) # Hidden State array to keep track in a vertical stack
        # print("HS State: ",hs[t].shape)
        ys.append(y) # Output State arrray to keep track in a vertical stack
    return hs, ys


# Compute cross-entropy loss and prepare for backpropagation
def cross_entropy_loss(y_pred,y_true_idx):
    """
    y_pred: (vocab_size, 1) probability vector after softmax
    y_true_idx: integer, index of the true next word
    """
    return -np.log(y_pred[y_true_idx,0] + 1e-9) # add epsilon to avoid log(0)



# Implementation of backpropagation

def rnn_backward(X_embedded, Y, hs, ys, Wx, Wh, b, Why, by, E, learning_rate=0.01):
    vocab_size = Why.shape[0]
    hidden_size = Wh.shape[0]
    seq_len = len(Y)

    # Initialize gradients
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dWhy = np.zeros_like(Why)
    dby = np.zeros_like(by)
    dE = np.zeros_like(E)


    dh_next = np.zeros((hidden_size,1))

    # Backprop through time (from last to first step)

    for t in reversed(range(seq_len)):
        # Softmax
        exp_scores = np.exp(ys[t])
        p_t = exp_scores / np.sum(exp_scores)


        # Output error
        dy = p_t - one_hot(Y[t],vocab_size) # shape (vocab_size,1)

        # Gradient for output layer
        dWhy += np.dot(dy,hs[t].T)
        dby += dy

        # Backprop into hidden state
        dh = np.dot(Why.T, dy) + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh  # tanh' = 1 - h^2


        # Gradients for hidden layer
        x_t = X_embedded[t].reshape(-1,1)
        dWx += np.dot(dh_raw,x_t.T)
        dWh += np.dot(dh_raw, hs[t-1].T if t > 0 else np.zeros_like(hs[t]).T)
        db += dh_raw


        # Gradient for embedding vector
        word_idx = np.argmax(X_embedded[t] @ E.T) # approximate lookup
        dE[word_idx] += np.dot(Wx.T, dh_raw).ravel()

        # Pass gradient backward in time
        dh_next = np.dot(Wh.T,dh_raw)
    
    # Gradient clipping
    for dparam in [dWx, dWh, db, dWhy, dby, dE]:
        np.clip(dparam, -5, 5, out=dparam)

    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh 
    b -= learning_rate * db
    Why -= learning_rate * dWhy
    by -= learning_rate * dby
    E -= learning_rate * dE


    return Wx, Wh, b, Why, by, E
