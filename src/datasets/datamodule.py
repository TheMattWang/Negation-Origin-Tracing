from lightning import LightningDataModule
from omegaconf import DictConfig
from transformers import AutoTokenizer

class NOTDataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig):
        pass
    
    def __len__(self):
        return self.(len)