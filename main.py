# 2025/01/08 created w/ Aladdin Persson tutorial
# 2025/01/15 updated w/ Ben Trevett tutorial

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import network
from text_dataset_loader import TextDataset, get_dataloader
import hyper_params as hp


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for sources, targets in data_loader:
        sources, targets = sources.to(device), targets.to(device)

        predictions = model(sources)
        loss = loss_fn(predictions, targets)  # calculate loss
        optimiser.zero_grad()  # reset gradient at each iteration to 0
        loss.backward()  # backpropagate loss
        optimiser.step()  # update the weights

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
    print("Training is done.")


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
    # preparing data and building vocabulary
    print(" - Preparing dataset:")
    annotation_file = "Dataset/English_txt_harmony.csv"
    text_dataset = TextDataset(annotation_file, hp.special_tokens)

    train_data, validation_data, test_data = random_split(text_dataset, [0.8, 0.1, 0.1])
    train_dataloader = get_dataloader(train_data)
    test_dataloader = get_dataloader(test_data)

    # define input output size
    encoder_input_size = len(text_dataset.ur_alphabet)
    decoder_input_size = len(text_dataset.sr_alphabet)
    output_size = len(text_dataset.sr_alphabet)

    # build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    writer = SummaryWriter(f'runs/Loss_plot')
    step = 0

    encoder_net = network.Encoder(encoder_input_size, hp.encoder_embedding_size,
                                  hp.hidden_size, hp.num_layers, hp.encoder_dropout).to(device)
    decoder_net = network.Decoder(decoder_input_size, hp.decoder_embedding_size,
                                  hp.hidden_size, output_size, hp.num_layers, hp.decoder_dropout).to(device)
    seq2seq = network.Seq2Seq(encoder_net, decoder_net).to(device)

    # instantiate loss function + optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=text_dataset.specials.index)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=hp.learning_rate)

    # train model

    train(seq2seq, train_dataloader, loss_fn, optimizer, device, hp.num_epochs)

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
