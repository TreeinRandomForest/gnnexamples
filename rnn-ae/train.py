import torch.nn as nn
import torch.optim as optim

def get_crit_opt(net):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)

    return criterion, optimizer

def train(model, 
          dl,  #dataloader
          optimizer, 
          criterion,
          n_epochs,
          print_freq=1):
    
    model.train()
    for i in range(n_epochs):

        total_loss = 0
        total_items = 0

        for idx, data in enumerate(dl):
            pred = model(data)

            loss = criterion(pred, 
                             data.y)
            
            total_loss += loss.item()
            total_items += data.x.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % print_freq == 0:
            print(f'Epoch = {i} Training Loss = {total_loss:.3f}')

    return model, optimizer
