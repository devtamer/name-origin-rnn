import torch
import torch.nn as nn
import random
from rnn2_utils import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

print('num of classes:', n_categories)
print('num of letters:', n_letters)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """
        x: size [seq_length, 1, input_size]
        """
        h = torch.zeros(x.size(1), self.hidden_size)

        for i in range(x.size(0)):
            h = self.rnn_cell(x[i,:,:], h)
        
        # Hint: first call fc, then call softmax
        out = self.softmax(self.fc(h))
        
        return out

# Evaluate Task 1
torch.manual_seed(0)
rnn = RNN(10, 20, 18)
input_data = torch.randn(6, 3, 10)

with torch.no_grad():
    out = rnn(input_data)
    
print(out.size())
print(out[0])

def train(model, n_iters = 100000, print_every = 5000, plot_every = 1000, learning_rate = 0.005):
    # Turn on the training model
    model.train()
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    running_loss = 0
    all_losses = []
    
    # Train loop
    start = time.time()
    for i in range(n_iters):
        y, x, y_tensor, x_tensor = randomTrainingExample()
        # zero grad
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        
        # Backprop and update
        loss.backward()
        optimizer.step()
        
        # Record loss
        running_loss += loss.item()
        
        # Print iter, loss, name, and guess
        if i % print_every == 0 and i > 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == y else '✗ (%s)' % y
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, timeSince(start), loss, x, guess, correct))
        # Append loss
        if i % plot_every == 0 and i > 0:
            all_losses.append(running_loss / plot_every)
            running_loss = 0
    print("\n \n")

    # Plot
    plt.figure()
    plt.plot(all_losses)

# Evaluate Task 2
# Be patient with the training speed :)
torch.manual_seed(0)
random.seed(0)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

train(rnn)


# class BiRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(BiRNN, self).__init__()
#         self.hidden_size = hidden_size
        
#         self.rnn_cell1 = nn.RNNCell(input_size, hidden_size)
#         self.rnn_cell2 = nn.RNNCell(input_size, hidden_size)
        
#         self.fc = nn.Linear(2 * hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
    
#     def forward(self, x):
#         """
#         x: size [seq_length, 1, input_size]
#         """
#         h1 = torch.zeros(x.size(1), self.hidden_size)
#         for i in range(x.size(0)):
#             h1 = self.rnn_cell1(x[i,:,:], h1)
        
#         h2 = torch.zeros(x.size(1), self.hidden_size)
#         for i in reversed(range(x.size(0))): 
#             h2 = self.rnn_cell2(x[i,:,:], h2)
        
#         h = torch.cat((h1, h2), dim=1)
#         out = self.softmax(self.fc(h))

        
#         return out

# # Evaluate Task 3
# # Be even more patient, as the training time is almost doubled :P
# torch.manual_seed(0)
# random.seed(0)

# n_hidden = 128
# birnn = BiRNN(n_letters, n_hidden, n_categories)

# train(birnn)