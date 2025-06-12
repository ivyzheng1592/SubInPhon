# created 2025/03/07 structure based on Ben Trevett tutorial
# updated 2025/04/01 upgraded to class
# A script that defines training and evaluation at each epoch of each run

from text_run import *


class FeatureRun(TextRun):
    def __init__(self, seq2seq, recorder, run_num):
        super().__init__(seq2seq, recorder, run_num)

    def initialize_weight(self):
        for name, param in self.seq2seq.named_parameters():
            if "weight" in name and "embedding" not in name:  # weights other than embedding weights
                nn.init.normal_(param.data, mean=0, std=0.01)
            elif "bias" in name:  # biases
                nn.init.constant_(param.data, 0)
