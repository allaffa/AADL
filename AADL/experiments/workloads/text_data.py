"""WikiText language modeling through the lazy Hugging Face datasets API."""

import math

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, partition_indices


class ByteTransformerLM(torch.nn.Module):
    def __init__(self, width=128, layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, width)
        block = torch.nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0.1, batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(block, layers)
        self.output = torch.nn.Linear(width, 256)

    def forward(self, tokens):
        length = tokens.size(1)
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), 1,
        )
        return self.output(self.encoder(self.embedding(tokens), mask=mask))


def _token_sequences(split, sequence_length, max_tokens):
    stream = bytearray()
    for text in split["text"]:
        stream.extend(text.encode("utf-8", errors="replace"))
        stream.append(10)
        if len(stream) >= max_tokens:
            break
    values = torch.tensor(list(stream[:max_tokens]), dtype=torch.long)
    usable = ((values.numel() - 1) // sequence_length) * sequence_length
    inputs = values[:usable].view(-1, sequence_length)
    targets = values[1:usable + 1].view(-1, sequence_length)
    return inputs, targets


class WikiTextWorkload:
    family = "transformer"
    name = "transformer.wikitext103"

    def __init__(self, values):
        options = Options(values)
        self.cache_dir = options.string("root", required=True)
        self.batch_size = options.integer("batch_size", 16)
        self.sequence_length = options.integer("sequence_length", 128)
        self.max_train_tokens = options.integer("max_train_tokens", 10_000_000)
        self.max_eval_tokens = options.integer("max_eval_tokens", 1_000_000)
        self.width = options.integer("width", 128)
        self.layers = options.integer("layers", 2)
        self.offline = options.boolean("offline", False)
        options.finish()

    def build(self, device, seed):
        try:
            from datasets import DownloadConfig, load_dataset
        except ImportError as error:
            raise ImportError(
                "WikiText requires Hugging Face datasets; install requirements-paper.txt"
            ) from error
        download = DownloadConfig(local_files_only=self.offline)
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1",
            cache_dir=self.cache_dir, download_config=download,
        )
        train_x, train_y = _token_sequences(
            dataset["train"], self.sequence_length, self.max_train_tokens,
        )
        valid_x, valid_y = _token_sequences(
            dataset["validation"], self.sequence_length, self.max_eval_tokens,
        )
        model = ByteTransformerLM(self.width, self.layers).to(device)

        def train_batches(epoch, rank, world_size):
            indices = partition_indices(
                torch.zeros(train_x.size(0)), rank, world_size, seed, epoch,
            )
            for group in indices.split(self.batch_size):
                if group.numel() == self.batch_size:
                    yield train_x[group].to(device), train_y[group].to(device)

        def loss(module, batch):
            logits = module(batch[0])
            return torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), batch[1].flatten(),
            )

        def evaluate(module):
            total_loss = total_tokens = 0
            module.eval()
            with torch.no_grad():
                for start in range(0, valid_x.size(0), self.batch_size):
                    x = valid_x[start:start + self.batch_size].to(device)
                    y = valid_y[start:start + self.batch_size].to(device)
                    if not x.numel():
                        continue
                    logits = module(x)
                    value = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1), y.flatten(), reduction="sum",
                    )
                    total_loss += float(value)
                    total_tokens += y.numel()
            average = total_loss / total_tokens
            return {"loss": average, "perplexity": math.exp(min(average, 50.0))}

        return WorkloadInstance(model, train_batches, loss, evaluate)


@register_workload(WikiTextWorkload.name)
def create(options):
    return WikiTextWorkload(options)


class GlueSST2Workload:
    family = "transformer"
    name = "transformer.glue-sst2"

    def __init__(self, values):
        options = Options(values)
        self.cache_dir = options.string("root", required=True)
        self.batch_size = options.integer("batch_size", 32)
        self.sequence_length = options.integer("sequence_length", 128)
        self.max_train_samples = options.integer("max_train_samples", 67_349)
        self.max_eval_samples = options.integer("max_eval_samples", 872)
        self.model_name = options.string("model_name", "distilbert-base-uncased")
        self.offline = options.boolean("offline", False)
        options.finish()

    def build(self, device, seed):
        try:
            from datasets import DownloadConfig, load_dataset
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "GLUE requires datasets and transformers; install requirements-paper.txt"
            ) from error
        dataset = load_dataset(
            "nyu-mll/glue", "sst2", cache_dir=self.cache_dir,
            download_config=DownloadConfig(local_files_only=self.offline),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir,
            local_files_only=self.offline,
        )

        def encode(split, maximum):
            rows = split.select(range(min(len(split), maximum)))
            encoded = tokenizer(
                rows["sentence"], max_length=self.sequence_length,
                truncation=True, padding="max_length", return_tensors="pt",
            )
            return (
                encoded["input_ids"], encoded["attention_mask"],
                torch.tensor(rows["label"], dtype=torch.long),
            )

        train_x, train_mask, train_y = encode(dataset["train"], self.max_train_samples)
        valid_x, valid_mask, valid_y = encode(dataset["validation"], self.max_eval_samples)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, cache_dir=self.cache_dir,
            local_files_only=self.offline,
        ).to(device)

        def train_batches(epoch, rank, world_size):
            indices = partition_indices(train_y, rank, world_size, seed, epoch)
            for group in indices.split(self.batch_size):
                if group.numel() == self.batch_size:
                    yield (
                        train_x[group].to(device), train_mask[group].to(device),
                        train_y[group].to(device),
                    )

        def loss(module, batch):
            logits = module(input_ids=batch[0], attention_mask=batch[1]).logits
            return torch.nn.functional.cross_entropy(logits, batch[2])

        def evaluate(module):
            correct = total = 0
            module.eval()
            with torch.no_grad():
                for start in range(0, valid_x.size(0), self.batch_size):
                    x = valid_x[start:start + self.batch_size].to(device)
                    mask = valid_mask[start:start + self.batch_size].to(device)
                    y = valid_y[start:start + self.batch_size].to(device)
                    logits = module(input_ids=x, attention_mask=mask).logits
                    correct += int((logits.argmax(-1) == y).sum())
                    total += y.numel()
            return {"accuracy": correct / total}

        return WorkloadInstance(model, train_batches, loss, evaluate)


@register_workload(GlueSST2Workload.name)
def create_glue(options):
    return GlueSST2Workload(options)
