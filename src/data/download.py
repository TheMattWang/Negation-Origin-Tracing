from datasets import load_dataset

dataset = load_dataset("stanfordnlp/sst2")
train = load_dataset("stanfordnlp/sst2",split="train")
validation = load_dataset("stanfordnlp/sst2",split="validation")
test = load_dataset("stanfordnlp/sst2",split="test")

train.to_parquet("/Users/mattwang/Documents/NLP/Negation-Origin-Tracing/data/raw/train/sst.parquet")
validation.to_parquet("/Users/mattwang/Documents/NLP/Negation-Origin-Tracing/data/raw/validation/sst.parquet")
test.to_parquet("/Users/mattwang/Documents/NLP/Negation-Origin-Tracing/data/raw/test/sst.parquet")

ds = load_dataset("ceval/contrastive-sentiment-negation")
train = load_dataset("ceval/contrastive-sentiment-negation",split="train")
validation = load_dataset("ceval/contrastive-sentiment-negation",split="validation")
test = load_dataset("ceval/contrastive-sentiment-negation",split="test")


train.to_parquet("/Users/mattwang/Documents/NLP/Negation-Origin-Tracing/data/raw/train/negation.parquet")
validation.to_parquet("/Users/mattwang/Documents/NLP/Negation-Origin-Tracing/data/raw/validation/negation.parquet")
test.to_parquet("/Users/mattwang/Documents/NLP/Negation-Origin-Tracing/data/raw/test/negation.parquet")
