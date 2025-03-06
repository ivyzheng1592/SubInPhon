import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.init import uniform_


# training loop within one epoch
def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()  # enable dropout in training
    epoch_loss = 0

    for i, (src, trg) in enumerate(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        optimizer.zero_grad()  # reset gradient at each iteration to 0
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg_length, batch_size, output_dim]

        # remove the <SOS> token from output and target and reshape for loss calculation
        output_dim = output.shape[2]
        output = output[1:].view(-1, output_dim)
        # output = [(trg_length - 1) * batch_size, output_dim]
        trg = trg[1:].view(-1)
        # trg = [(trg_length - 1) * batch_size]

        loss = criterion(output, trg)  # calculate loss
        loss.backward()  # backpropagate loss
        clip_grad_norm_(model.parameters(), clip) # clip the gradients to prevent exploding
        optimizer.step()  # update the weights

        epoch_loss += loss.item()  # summarize the loss value

    average_loss = epoch_loss / len(data_loader)  # average the loss value over all batches

    return average_loss


# evaluation loop within one epoch
def evaluate_fn(model, data_loader, criterion, device):
    model.eval()  # disable dropout in evaluation
    epoch_loss = 0
    with torch.no_grad():  # disable gradient tracking
        for i, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            # src = [src_len, batch_size]
            # trg = [trg_len, batch_size]

            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg_len, batch_size, output_dim]

            # remove the <SOS> token from output and target and reshape for loss calculation
            output_dim = output.shape[2]
            output = output[1:].view(-1, output_dim)
            # output = [(trg_len - 1) * batch_size, output_dim]
            trg = trg[1:].view(-1)
            # trg = [(trg_len - 1) * batch_size, output_dim]

            loss = criterion(output, trg)  # calculate loss
            epoch_loss += loss.item()  # summarize the loss value

        average_loss = epoch_loss / len(data_loader)  # average the loss value over all batches

        return average_loss


# weight initialization
def init_weights(model):
    for name, param in model.named_parameters():
        uniform_(param.data, -0.08, 0.08)

