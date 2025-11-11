from datasets import load_dataset

dataset = load_dataset("stanfordnlp/sst2")
train = load_dataset("stanfordnlp/sst2",split="train")
validation = load_dataset("stanfordnlp/sst2",split="validation")
test = load_dataset("stanfordnlp/sst2",split="test")

train.to_parquet("path")
validation.to_parquet("path")
test.to_parquet("path")
