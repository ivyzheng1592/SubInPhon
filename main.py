# created 2025/03/03 w/ Ben Trevett tutorial

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torchsummary import summary
import network
from hyper_params import pad_token
from text_dataset_loader import TextDataset, get_dataloader
import hyper_params as hp


# training loop
def train_fn(model, data_loader, optimizer, loss_fn, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        # src = [src_length, batch_size]
        # trg = [trg_length, batch_size]

        optimizer.zero_grad()  # reset gradient at each iteration to 0

        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [trg_length, batch_size, output_dim]
        # -> output = [(trg_length - 1) * batch_size, output_dim]

        trg = trg[1:].view(-1)
        # trg = [trg_length, batch_size]
        # -> trg = [(trg_length - 1) * batch_size]

        loss = loss_fn(output, trg)  # calculate loss
        loss.backward()  # backpropagate loss
        nn.utils.clip_grad_norm(model.parameters(), clip) # clip the gradients to prevent exploding
        optimizer.step()  # update the weights
        epoch_loss += loss.item()  # summarize the loss value
        average_loss = epoch_loss / len(data_loader)  # average the loss value over all batches

    return average_loss


def predict(model, source, target, class_mapping):
    model.eval()
    with torch.no_grad(): # context manager
        predictions = model(source)
        # Tensor (length of input, number of classes) -> [[0.1, 0.01, ..., 0.6]]
        predicted_index = predictions[0].argmax(0)  # get the highest predicted value
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


def main():
    print(" - Loading dataset and building vocabulary:")
    annotations_file = "Dataset/English_txt_harmony.csv"
    text_dataset = TextDataset(annotations_file, hp.special_tokens)
    train_data, validation_data, test_data = random_split(text_dataset, [0.8, 0.1, 0.1])

    print(" - Creating dataloader:")
    train_dataloader = get_dataloader(train_data)
    test_dataloader = get_dataloader(test_data)

    print(" - Initializing model:")
    # model hyperparameters
    encoder_input_dim = len(text_dataset.ur_alphabet)
    decoder_input_dim = len(text_dataset.sr_alphabet)
    output_dim = len(text_dataset.sr_alphabet)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # networks
    encoder_net = network.Encoder(encoder_input_dim, hp.encoder_embedding_dim,
                                  hp.hidden_dim, hp.n_layers, hp.encoder_dropout).to(device)
    decoder_net = network.Decoder(decoder_input_dim, hp.decoder_embedding_dim,
                                  hp.hidden_dim, output_dim, hp.n_layers, hp.decoder_dropout).to(device)
    seq2seq = network.Seq2Seq(encoder_net, decoder_net).to(device)
    summary(seq2seq, encoder_input_dim, hp.batch_size)

    # optimizer and loss function
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=hp.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=text_dataset.specials.index[pad_token])

    #writer = SummaryWriter(f'runs/Loss_plot')
    #step = 0

    print(" - Training model:")
    # train the model
    train(seq2seq, train_dataloader, loss_fn, optimizer, device, hp.n_epochs)

    # save the model
    torch.save(seq2seq.state_dict(), "seq2seq.pth")
    print("Model trained and stored at seq2seq.pth.")

    # load the model
    state_dict = torch.load("seq2seq.pth")
    seq2seq.load_state_dict(state_dict)

    # get a sample from the validation dataset for inference
    source, target = validation_data[0][0], validation_data[0][1]

    # make an inference
    predicted, expected = predict(seq2seq, source, target)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
