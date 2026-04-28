# created 2025/11/10
# A class that defines training and evaluation at each epoch of each run
# Data is recorded into dictionary in AudioRecorder

import os
from typing import Any, Optional, List, Tuple
import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
import re
import hyper_params as hp


class AudioTrainer:
    def __init__(self, seq2seq: Any, recorder: Any, resume_model_file: Optional[str] = None) -> None:

        # condition hyperparameters
        self.seq2seq = seq2seq
        self.recorder = recorder
        self.start_epoch = 0

        if resume_model_file:
            self.seq2seq.load_state_dict(torch.load(resume_model_file))
            match = re.search(r"_epoch(-?\d+)_seq2seq\.pth$", resume_model_file)
            if not match:
                raise RuntimeError(f"Cannot parse epoch from model file: {resume_model_file}")
            self.start_epoch = int(match.group(1)) + 1
            print(f"Resuming from {resume_model_file} at epoch {self.start_epoch}")

        # optimizer
        self.optimizer = torch.optim.Adam(self.seq2seq.parameters(), lr=hp.learning_rate)

    # a function for loss calculation
    def compute_loss(
        self,
        output: torch.Tensor,
        spec: torch.Tensor,
        trg_txt: torch.Tensor,
        trg_aud: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # remove the <SOS> token from output and target and reshape for loss calculation
        txt_dim = output.shape[2]
        output = output[1:].view(-1, txt_dim)
        # output = [(trg_len - 1) * batch_size, txt_dim]
        trg_txt = trg_txt[1:].view(-1)
        # trg = [(trg_len - 1) * batch_size]

        # detect padded 0s from target spectrogram for loss calculation
        weight = torch.where(torch.eq(trg_aud, -100), 0.0, 1.0)
        
        rec_loss = F.l1_loss(spec, trg_aud, reduction='mean', weight=weight)
        pred_loss = F.cross_entropy(output, trg_txt, ignore_index=hp.special_tokens.index(hp.pad_token))

        return rec_loss, pred_loss

    # a function that completes one repetition of training and evaluation
    def run(
        self,
        train_dataloader: Any,
        eval_dataloader: Any,
        eval_record_type: str,
        gen_eval_dataloader: Any = None,
    ) -> None:
        if self.start_epoch == 0:
            # save untrained model
            model_file = os.path.join(self.recorder.model_dir, self.recorder.run_root + "_epoch-1_seq2seq.pth")
            torch.save(self.seq2seq.state_dict(), model_file)
            print(f"Untrained model stored at {model_file}")

        # at each epoch, display progress bar
        for epoch in tqdm.tqdm(range(self.start_epoch, hp.n_epochs)):
            # update loss
            train_rec_loss, train_pred_loss, train_src, train_trg, train_pred = (
                self.train_one_epoch(train_dataloader, hp.text_teacher_forcing, hp.audio_teacher_forcing))
            eval_rec_loss, eval_pred_loss, eval_src, eval_trg, eval_pred = (
                self.evaluate_one_epoch(eval_dataloader))

            # record predictions and prediction correctness
            train_acc = self.recorder.record_pred(epoch, "train", train_src, train_trg, train_pred)
            eval_acc = self.recorder.record_pred(epoch, eval_record_type, eval_src, eval_trg, eval_pred)
            # record accuracy
            self.recorder.record_acc(epoch, "train", train_rec_loss, train_pred_loss, train_acc)
            self.recorder.record_acc(epoch, eval_record_type, eval_rec_loss, eval_pred_loss, eval_acc)
            if gen_eval_dataloader is not None:
                gen_eval_rec_loss, gen_eval_pred_loss, gen_eval_src, gen_eval_trg, gen_eval_pred = (
                    self.evaluate_one_epoch(gen_eval_dataloader)
                )
                gen_eval_acc = self.recorder.record_pred(
                    epoch,
                    "gen",
                    gen_eval_src,
                    gen_eval_trg,
                    gen_eval_pred,
                )
                self.recorder.record_acc(
                    epoch, 
                    "gen", 
                    gen_eval_rec_loss, 
                    gen_eval_pred_loss, 
                    gen_eval_acc
                )

            print(f"Epoch {epoch} Train Reconstruction Task Loss: {train_rec_loss:7.3f} "
                  f"| Train Prediction Task Loss: {train_pred_loss:7.3f} "
                  f"| Train Prediction Acc: {train_acc:7.3f}")
            print(f"Epoch {epoch} {eval_record_type.capitalize()} Reconstruction Task Loss: {eval_rec_loss:7.3f} "
                  f"| {eval_record_type.capitalize()} Prediction Task Loss: {eval_pred_loss:7.3f} "
                  f"| {eval_record_type.capitalize()} Prediction Acc: {eval_acc:7.3f}")
            if gen_eval_dataloader is not None:
                print(f"Epoch {epoch} Gen Reconstruction Task Loss: {gen_eval_rec_loss:7.3f} "
                      f"| Gen Prediction Task Loss: {gen_eval_pred_loss:7.3f} "
                      f"| Gen Prediction Acc: {gen_eval_acc:7.3f}")

            # save model every other save_epochs
            if epoch % hp.save_epochs == 0 or epoch == hp.n_epochs-1:
                model_file = os.path.join(
                    self.recorder.model_dir,
                    self.recorder.run_root + "_epoch" + str(epoch) + "_seq2seq.pth",
                )
                torch.save(self.seq2seq.state_dict(), model_file)
                print(f"Epoch {epoch} model trained and stored at {model_file}")

        # plot accuracy at the end of training
        utils.plot_aud_acc(self.recorder.acc_store, self.recorder.acc_plot)
        utils.save_to_file(self.recorder.acc_store, self.recorder.acc_file)
        utils.save_to_file(self.recorder.pred_store, self.recorder.pred_file)
        print(f"Run {self.recorder.run_num} training loss, accuracy, and predicted results are saved")


    # a function that manages training at one epoch
    def train_one_epoch(
        self,
        data_loader: Any,
        txt_teacher_forcing: float,
        aud_teacher_forcing: float,
    ) -> Tuple[float, float, List[Any], List[Any], List[Any]]:
        self.seq2seq.train()  # enable dropout in training
        epoch_pred_loss = 0
        epoch_rec_loss = 0

        # storing text predictions
        src_txts = []
        trg_txts = []
        pred_txts = []

        # training in one batch
        for i, input in enumerate(data_loader):
            src_txt, _, trg_txt, trg_aud = input
            # src_txt = [txt_src_len, batch_size]
            # src_aud = [batch_size, n_channels, n_freq, aud_src_len]
            # trg_txt = [txt_trg_len, batch_size]
            # trg_aud = [batch_size, n_channels, n_freq, aud_trg_len]

            self.optimizer.zero_grad()  # reset gradient at each iteration to 0
            output, pred, spec, _, _ = self.seq2seq(input, txt_teacher_forcing, aud_teacher_forcing)
            # output = [txt_trg_len, batch_size, txt_output_dim]
            # pred = [txt_trg_len, batch_size]
            # spec = [batch_size, 1, aud_output_dim, aud_trg_len]

            # record predictions
            src_txts.append(src_txt)
            trg_txts.append(trg_txt)
            pred_txts.append(pred)

            rec_loss, pred_loss = self.compute_loss(output, spec, trg_txt, trg_aud)
            batch_loss = rec_loss + pred_loss  # calculate batch loss
            epoch_rec_loss += rec_loss.item()
            epoch_pred_loss += pred_loss.item()  # add to epoch loss
            batch_loss.backward()  # backpropagate loss
            nn.utils.clip_grad_norm_(self.seq2seq.parameters(), max_norm=1.0)
            # clip the gradients to prevent exploding, uncomment if necessary
            self.optimizer.step()  # update the weights

        # average loss over all batches
        epoch_rec_loss = epoch_rec_loss / len(data_loader)
        epoch_pred_loss = epoch_pred_loss / len(data_loader)
        return epoch_rec_loss, epoch_pred_loss, src_txts, trg_txts, pred_txts

    # a function that manages evaluation at one epoch
    def evaluate_one_epoch(self, data_loader: Any) -> Tuple[float, float, List[Any], List[Any], List[Any]]:
        self.seq2seq.eval()  # disable dropout in evaluation
        epoch_rec_loss = 0
        epoch_pred_loss = 0

        # storing text predictions
        src_txts = []
        trg_txts = []
        pred_txts = []

        # evaluation in one batch
        with torch.no_grad():  # disable gradient tracking
            for i, input in enumerate(data_loader):
                src_txt, _, trg_txt, trg_aud = input
                # src_txt = [txt_src_len, batch_size]
                # src_aud = [batch_size, n_channels, n_freq, aud_src_len]
                # trg_txt = [txt_trg_len, batch_size]
                # trg_aud = [batch_size, n_channels, n_freq, aud_trg_len]

                output, pred, spec, _, _ = self.seq2seq(input, 0, 0)  # turn off teacher forcing
                # output = [txt_trg_len, batch_size, txt_output_dim]
                # pred = [txt_trg_len, batch_size]
                # spec = [batch_size, 1, aud_output_dim, aud_trg_len]

                # record predictions
                src_txts.append(src_txt)
                trg_txts.append(trg_txt)
                pred_txts.append(pred)

                rec_loss, pred_loss = self.compute_loss(output, spec, trg_txt, trg_aud)  # calculate batch loss
                epoch_rec_loss += rec_loss.item()
                epoch_pred_loss += pred_loss.item()  # add to epoch loss

        # average loss and accuracy over all batches
        epoch_rec_loss = epoch_rec_loss / len(data_loader)
        epoch_pred_loss = epoch_pred_loss / len(data_loader)
        return epoch_rec_loss, epoch_pred_loss, src_txts, trg_txts, pred_txts

    # a function that manages evaluation of one random batch
    def evaluate_attention(
        self,
        dataloader: Any,
        eval_epoch: int = hp.n_epochs-1,
    ) -> None:
        # get one random batch of test data
        dataiter = iter(dataloader)
        input = next(dataiter)

        # load model
        model_file = os.path.join(
            self.recorder.model_dir,
            self.recorder.run_root + "_epoch" + str(eval_epoch) + "_seq2seq.pth",
        )
        self.seq2seq.load_state_dict(torch.load(model_file))
        self.seq2seq.eval()  # disable dropout in evaluation

        with torch.no_grad():  # disable gradient tracking
            # get predicted sr and attention weights
            _, pred, spec, txt_atts, aud_atts = self.seq2seq(input, 0, 0)  # turn off teacher forcing
            # pred = [txt_trg_len, batch_size]
            # spec = [batch_size, 1, aud_output_dim, aud_trg_len]
            # txt_atts = [txt_trg_len, batch_size, aud_src_len]
            # aud_atts = [aud_trg_len, batch_size, aud_src_len]

            src_txt, src_aud, trg_txt, _ = input
            # src_txt = [txt_src_len, batch_size]
            # src_aud = [batch_size, n_channels, n_freq, aud_src_len]
            # trg_txt = [txt_trg_len, batch_size]
            # trg_aud = [batch_size, n_channels, n_freq, aud_trg_len]

            for i in range(hp.batch_size):
                ur_txt = src_txt[:, i]
                sr_txt = trg_txt[:, i]
                pred_sr_txt = pred[:, i]

                # convert predictions
                ur_string, _, pred_sr_string = self.recorder.tensor2string(ur_txt, sr_txt, pred_sr_txt)
                _, _, pred_sr_list = self.recorder.tensor2list(ur_txt, sr_txt, pred_sr_txt)

                # retrieve spectrograms and attention weights
                ur_spec = src_aud[i, :, :, :][0]
                # ur_spec = [n_freq, dur]
                pred_sr_spec = spec[i, :, :, :][0]
                # pred_sr_spec = [n_freq, dur]
                txt_att = txt_atts[:, i, :]
                # txt_att = [txt_trg_len, aud_src_len]
                aud_att = aud_atts[:, i, :]
                # aud_att = [aud_trg_len, aud_src_len]

                # convert spectrograms
                # include only the parts of spectrogram that not paddings
                non_zeros = self.recorder.dataset.remove_padding(ur_spec)
                ur_spec = ur_spec[:, non_zeros]
                pred_sr_spec = pred_sr_spec[:, non_zeros]
                txt_att = txt_att[:, non_zeros]
                aud_att = aud_att[non_zeros, :][:, non_zeros]

                # plot attention
                txt_att_plot = os.path.join(
                    self.recorder.att_plot_dir,
                    self.recorder.run_root + "_epoch" + str(eval_epoch) + "_" + ur_string + "_" + pred_sr_string + "_txt.png"
                )
                aud_att_plot = os.path.join(
                    self.recorder.att_plot_dir,
                    self.recorder.run_root + "_epoch" + str(eval_epoch) + "_" + ur_string + "_" + pred_sr_string + "_aud.png"
                )
                utils.plot_aud_att(ur_spec, pred_sr_list, pred_sr_spec, txt_att, aud_att,
                                   txt_att_plot, aud_att_plot)
            print(f"Run {self.recorder.run_num} attention plots are saved for investigation")

    def evaluate_embedding(self) -> None:
        # a dictionary of dictionaries to store all embeddings
        phone_spaces = {}
        # select focus group
        focus = list(self.recorder.language.focus.keys())

        for file_name in os.listdir(self.recorder.model_dir):
            # load model
            model_file = os.path.join(self.recorder.model_dir, file_name)
            self.seq2seq.load_state_dict(torch.load(model_file))

            # retrieve target embedding
            embed = self.seq2seq.decoder.embedding.weight

            # retrieve embedding of all phonemes and focus group
            phone_space = self.recorder.dataset.sr_alphabet.embed2fea(embed)
            phone_spaces[file_name] = phone_space

            embed_file = os.path.join(self.recorder.embed_plot_dir,
                                      file_name.replace("_seq2seq.pth", "_embedding.csv"))
            embed_plot = os.path.join(self.recorder.embed_plot_dir,
                                      file_name.replace("_seq2seq.pth", "_embedding.png"))
            # plot embedding
            utils.plot_embed(phone_space, focus, embed_plot)
            # save embedding recording to file
            utils.save_to_file(phone_space, embed_file)

        # plot embedding
        utils.plot_embed_updated(phone_spaces, focus, self.recorder.embed_plot, self.recorder.focus_embed_plot)
        print(f"Run {self.recorder.run_num} embedding plots and files are saved for investigation")

    def evaluate_audio_embedding(
        self,
        dataloader: Any,
        eval_epoch: int = hp.n_epochs - 1,
    ) -> None:
        # Load the saved model checkpoint used for audio embedding inspection.
        model_file = os.path.join(
            self.recorder.model_dir,
            self.recorder.run_root + "_epoch" + str(eval_epoch) + "_seq2seq.pth"
        )
        self.seq2seq.load_state_dict(torch.load(model_file))
        self.seq2seq.eval()

        # Run the model on a small number of evaluation batches and extract vowel embeddings.
        with torch.no_grad():
            for i, input in enumerate(dataloader):
                if i >= hp.aud_embed_inspect_batch:
                    break
                src_txt, src_aud, trg_txt, trg_aud = input
                _, pred_txt, pred_spec, _, _ = self.seq2seq(input, 0, 0)

                # Recover the word strings for direct audio-reference lookup.
                for j in range(hp.batch_size):
                    ur_string, sr_string, _ = self.recorder.tensor2string(
                        src_txt[:, j],
                        trg_txt[:, j],
                        pred_txt[:, j],
                    )
                    ur_ref = self.recorder.dataset.word_to_ur_ref(ur_string)
                    sr_ref = self.recorder.dataset.word_to_sr_ref(sr_string)

                    # Load the source and target TextGrid files for vowel segmentation.
                    textgrid_dir = os.path.join(hp.audio_root, hp.lang_name + "_segmented")
                    src_textgrid = os.path.join(textgrid_dir, ur_ref + ".TextGrid")
                    trg_textgrid = os.path.join(textgrid_dir, sr_ref + ".TextGrid")
                    if not os.path.exists(src_textgrid):
                        raise FileNotFoundError(f"Could not find TextGrid for {ur_ref} in {textgrid_dir}")
                    if not os.path.exists(trg_textgrid):
                        raise FileNotFoundError(f"Could not find TextGrid for {sr_ref} in {textgrid_dir}")

                    # Record one row per vowel token into the three embedding stores.
                    self.recorder.record_audio_embedding("source", src_aud[j, 0], src_textgrid, ur_ref)
                    self.recorder.record_audio_embedding("target", trg_aud[j, 0], trg_textgrid, sr_ref)
                    self.recorder.record_audio_embedding("pred", pred_spec[j, 0], trg_textgrid, sr_ref)

        # Save the three embedding stores to CSV files.
        utils.save_to_file(self.recorder.source_audio_embed_store, self.recorder.source_audio_embed_file)
        utils.save_to_file(self.recorder.target_audio_embed_store, self.recorder.target_audio_embed_file)
        utils.save_to_file(self.recorder.pred_audio_embed_store, self.recorder.pred_audio_embed_file)

        # Plot the source, target, and predicted vowel embeddings.
        utils.plot_aud_embed(
            self.recorder.source_audio_embed_store,
            self.recorder.source_audio_embed_plot,
        )
        utils.plot_aud_embed(
            self.recorder.target_audio_embed_store,
            self.recorder.target_audio_embed_plot,
        )
        utils.plot_aud_embed(
            self.recorder.pred_audio_embed_store,
            self.recorder.pred_audio_embed_plot,
        )
        print(
            f"Run {self.recorder.run_num} source, target, and predicted audio embedding plots are saved "
            f"({len(self.recorder.source_audio_embed_store['vowel_label'])}, "
            f"{len(self.recorder.target_audio_embed_store['vowel_label'])}, "
            f"{len(self.recorder.pred_audio_embed_store['vowel_label'])} vowel tokens)"
        )
