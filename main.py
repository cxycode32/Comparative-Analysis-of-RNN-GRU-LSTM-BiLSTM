import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time


class BaseRNN(nn.Module):
    def __init__(self, cell_type, input_size, sequence_length, hidden_size, num_layers, num_classes, learning_rate, batch_size, num_epochs, bidirectional=False):
        super(BaseRNN, self).__init__()
        self.cell_type = cell_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("cell_type must be 'rnn', 'gru', or 'lstm'")

        self.fc = nn.Linear(hidden_size * direction_factor * sequence_length, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(x.device)

        if self.cell_type == "lstm":
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def load_data(batch_size):
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader, device, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_accuracies = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for data, targets in train_loader:
            data = data.to(device).squeeze(1)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        accuracy = check_accuracy(test_loader, model, device)
        test_accuracies.append(accuracy)

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    return train_losses, test_accuracies


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    accuracy = 100 * num_correct / num_samples
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def plot_results(train_losses, test_accuracies, labels):
    for i, label in enumerate(labels):
        plt.plot(train_losses[i], label=f"{label} Train Loss")
        plt.plot(test_accuracies[i], label=f"{label} Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.title("Model Comparison: RNN vs GRU vs LSTM vs BiLSTM")
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyperparams = {
        'input_size': 28,
        'sequence_length': 28,
        'hidden_size': 256,
        'num_layers': 2,
        'num_classes': 10,
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 2
    }

    train_loader, test_loader = load_data(hyperparams['batch_size'])

    models = {
        "RNN": BaseRNN("rnn", **hyperparams).to(device),
        "GRU": BaseRNN("gru", **hyperparams).to(device),
        "LSTM": BaseRNN("lstm", **hyperparams).to(device),
        "BiLSTM": BaseRNN("lstm", bidirectional=True, **hyperparams).to(device),
    }

    train_losses_list, test_accuracies_list = [], []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        train_losses, test_accuracies = train_and_evaluate(
            model, train_loader, test_loader, device, hyperparams['num_epochs'], hyperparams['learning_rate']
        )
        train_losses_list.append(train_losses)
        test_accuracies_list.append(test_accuracies)

    plot_results(train_losses_list, test_accuracies_list, list(models.keys()))


if __name__ == "__main__":
    main()
