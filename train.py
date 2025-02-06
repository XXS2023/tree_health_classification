import torch
import torch.optim as optim
import torch.nn.functional as F
from model import TreeHealthModel


def train_model(X_train, y_train, input_size, epochs=100):
    model = TreeHealthModel(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))

        loss = F.cross_entropy(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    return model