import torch

def predict(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X))
        _, predicted = torch.max(outputs.data, 1)
    return predicted.numpy()