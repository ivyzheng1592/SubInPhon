import os
import tqdm
import numpy as np

from text_dataset_loader import TextDataset
from audio_dataset_loader import AudioDataset
from network import *
from run_setup import *
import hyper_params as hp

# a function that completes one run of training and evaluation of one model
def run_once(seq2seq, train_dataloader, valid_dataloader, test_dataloader,
             trial_num, language, datatype, condition, run):

    # results files
    acc_file = os.path.join("Results", trial_num,
                            language + "_" + datatype + "_" + condition + "_run" + str(run) + "_acc.csv")
    model_file = os.path.join("Results", trial_num,
                              language + "_" + datatype + "_" + condition + "_run" + str(run) + "_seq2seq.pth")
    acc_plot = os.path.join("Results", trial_num,
                            language + "_" + datatype + "_" + condition + "_run" + str(run) + "_acc_plot.png")

    # model weight initialization
    seq2seq.apply(init_weights)
    model_parameters = sum(p.numel() for p in seq2seq.parameters() if p.requires_grad)
    print(f"The model has {model_parameters} trainable parameters")

    # optimizer and loss function
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=hp.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=hp.special_tokens.index(hp.pad_token))

    # at each epoch, display progress bar
    for epoch in tqdm.tqdm(range(hp.n_epochs)):
        # update loss for each batch
        train_loss, train_acc = train_one_epoch(seq2seq, train_dataloader, optimizer, criterion,
                                                hp.clip, hp.teacher_forcing_ratio)
        valid_loss, valid_acc = evaluate_one_epoch(seq2seq, valid_dataloader, criterion)
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_acc:7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} | Valid Acc: {valid_acc:7.3f}")

        # save the accuracy value
        
        record_acc(acc_file, language, datatype, condition, run, epoch,
                   "train", train_loss, train_acc)
        record_acc(acc_file, language, datatype, condition, run, epoch,
                   "valid", valid_loss, valid_acc)
        print(f"Accuracy data saved at {acc_file}")

        # save the model
        torch.save(seq2seq.state_dict(), model_file)
        print(f"Model trained and stored at {model_file}")

    # plot the accuracy value
    plot_acc(acc_file, acc_plot)
    print(f"Accuracy plot save at {acc_plot}")

    # load the model
    seq2seq.load_state_dict(torch.load(model_file))

    # check loss for the test dataset
    test_loss, test_acc = evaluate_one_epoch(seq2seq, test_dataloader, criterion)
    print(f"\tTest Loss: {test_loss:7.3f} | Test PPL: {np.exp(test_loss):7.3f} | Test Acc: {test_acc:7.3f}")
    # save the accuracy value
    record_acc(acc_file, language, datatype, condition, run, hp.n_epochs,
               "test", test_loss, test_acc)
    print(f"Accuracy data saved at {acc_file}")


# a function that loads dataset and initializes model
# and completes multiple runs of training and evaluation of one model
def run_one_condition(trial_num, language, datatype, condition, n_runs, device):

    print(" - Loading dataset and building vocabulary:")
    annotations_file = os.path.join("Dataset", language + "_" + datatype + "_" + condition + ".csv")
    audio_dir = os.path.join("Dataset", "audio", language)

    if datatype == "txt":
        dataset = TextDataset(annotations_file, hp.special_tokens, device=device)
        print(f"The dataset contains {len(dataset)} UR-SR pairs")

        ur_vocab_size = len(dataset.ur_alphabet)
        sr_vocab_size = len(dataset.sr_alphabet)
        print(f"The UR vocabulary size is {ur_vocab_size}")
        print(f"The SR vocabulary size is {sr_vocab_size}")
    elif datatype == "aud":
        dataset = AudioDataset(annotations_file, audio_dir, hp.sample_rate, hp.n_samples,
                               hp.n_fft, hp.hop_length, hp.n_mels,
                               wav2mel=True, power2db=True, device=device)
        print(f"The dataset contains {len(dataset)} UR-SR pairs")

        ur, sr = dataset[0]
        ur_shape = ur.shape
        sr_shape = sr.shape
        print(f"The UR input shape is {ur_shape}")
        print(f"The SR input shape is {sr_shape}")
    else:
        raise ValueError("Invalid datatype input")

    print(" - Splitting dataset:")
    train_data, valid_data, test_data = dataset.split_dataset(hp.data_split_ratio)

    print(" - Creating dataloader:")
    train_dataloader = train_data.dataset.get_dataloader(hp.batch_size)
    valid_dataloader = valid_data.dataset.get_dataloader(hp.batch_size)
    test_dataloader = test_data.dataset.get_dataloader(hp.batch_size)

    print(" - Initializing model:")
    # model hyperparameters
    if datatype == "txt":
        encoder_input_dim = ur_vocab_size
        decoder_input_dim = sr_vocab_size
        output_dim = sr_vocab_size
    elif datatype == "aud":
        encoder_input_dim = ur.shape[1] * ur.shape[2]  # audio_freq * audio_dur
        decoder_input_dim = sr.shape[1] * sr.shape[2]
        output_dim = sr.shape[1] * sr.shape[2]
    else:
        raise ValueError("Invalid datatype input")

    # model initialization
    attention = BahdanauAttention(hp.hidden_dim)
    encoder_net = Encoder(encoder_input_dim, hp.encoder_embedding_dim, hp.hidden_dim,
                          hp.n_layers, hp.encoder_dropout).to(device)
    decoder_net = Decoder(decoder_input_dim, hp.decoder_embedding_dim, hp.hidden_dim, output_dim,
                          hp.n_layers, hp.decoder_dropout, attention).to(device)
    seq2seq = Seq2Seq(encoder_net, decoder_net, device).to(device)

    print(" - Training and evaluating model:")
    for run in n_runs:
        run_once(seq2seq, train_dataloader, valid_dataloader, test_dataloader,
                 trial_num, language, datatype, condition, run)


def inspect_one_condition(trial_num, language, datatype, condition, run):
    pass


if __name__ == "__main__":

    # defining the current trial of running
    trial_num = "250331"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # running each condition for x times
    languages = ["English"]
    datatypes = ["txt"]
    conditions = ["harmony", "disharmony"]
    n_runs = range(6, 7)
    for language in languages:
        for datatype in datatypes:
            for condition in conditions:
                run_one_condition(trial_num, language, datatype, condition, n_runs, device)