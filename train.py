import os
import pathlib
import typing

import torch
from google.protobuf import text_format

from myrtlespeech.model.deep_speech_2 import DeepSpeech2
from myrtlespeech.model.speech_to_text import SpeechToText
from myrtlespeech.run.callbacks.csv_logger import CSVLogger
from myrtlespeech.run.callbacks.callback import Callback, ModelCallback
from myrtlespeech.run.callbacks.clip_grad_norm import ClipGradNorm
from myrtlespeech.run.callbacks.report_mean_batch_loss import ReportMeanBatchLoss
from myrtlespeech.run.callbacks.stop_epoch_after import StopEpochAfter
from myrtlespeech.run.callbacks.mixed_precision import MixedPrecision
from myrtlespeech.post_process.utils import levenshtein
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.builders.task_config import build
from myrtlespeech.run.train import fit
from myrtlespeech.protos import task_config_pb2
from myrtlespeech.run.stage import Stage

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.backends.cudnn.benchmark = False

from myrtlespeech.model.cnn import MaskConv1d, MaskConv2d, PaddingMode

# parse example config file
with open("../src/myrtlespeech/configs/deep_speech_2_en.config") as f:
    task_config = text_format.Merge(f.read(), task_config_pb2.TaskConfig())

# create all components for config
seq_to_seq, epochs, train_loader, eval_loader = build(task_config)

class Profiler(Callback):
    """
    
    nvprof -f --profile-from-start off -o trace.nvvp -- python3 script.py
    
    
    Read using NVIDIA Visual Profiler (nvvp)
    
    """
    def on_batch_begin(self, *args, **kwargs):
        if not self.training:
            return
        if not (30 <= kwargs["total_train_batches"] <= 50):
            return
        self.prof = torch.autograd.profiler.emit_nvtx().__enter__()
        torch.cuda.profiler.start()
        
    def on_batch_end(self, **kwargs):
        if not self.training:
            return
        if not (30 <= kwargs["total_train_batches"] <= 50):
            return
        torch.cuda.profiler.stop()
        self.prof.__exit__(None, None, None)

from myrtlespeech.run.run import ReportCTCDecoder
from myrtlespeech.run.run import Saver
from myrtlespeech.run.run import TensorBoardLogger
from myrtlespeech.run.run import WordSegmentor

seq_to_seq.model
log_dir = "/home/samgd/logs/ds2"
# train the model
fit(
    seq_to_seq, 
    1000,#epochs, 
    train_loader=train_loader, 
    eval_loader=eval_loader,
    callbacks=[
        #prof,
        ReportMeanBatchLoss(),
        ReportCTCDecoder(
            seq_to_seq.post_process, 
            seq_to_seq.alphabet,
            WordSegmentor(" "),
        ),
        TensorBoardLogger(seq_to_seq.model, histograms=False),
        MixedPrecision(seq_to_seq, opt_level="O1"),
        #ClipGradNorm(seq_to_seq, max_norm=400),
        #StopEpochAfter(epoch_batches=1),
        CSVLogger(f"{log_dir}/log.csv", 
            exclude=[
                "epochs", 
                #"reports/CTCGreedyDecoder/transcripts",
            ]
        ),
        Saver(log_dir, seq_to_seq),
    ],
)
