import time
import lightning as L
import torch
from hydra.utils import instantiate


from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BaseModule(L.LightningModule):
    def __init__(
            self, 
            cfg: DictConfig,
            **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased")
        #
        self.cfg = cfg
    def forward(self,batch):
        pass

