import torch.nn as nn
# Define the model architecture
class MaskedLanguageModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskedLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        embedded = self.embedding(input)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        output = output[:,-1,:]
        # print(output.shape)
        output = self.softmax(output)
        # print(output.shape)
        return output
