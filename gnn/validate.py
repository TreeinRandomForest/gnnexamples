import torch

def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        output = model(data)
        predicted_labels = output > 0.5  # Threshold the output for binary predictions
        accuracy = (predicted_labels == data.y).all(dim=1).float().mean()
    
    return accuracy.item()