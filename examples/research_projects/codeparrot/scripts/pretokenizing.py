import multiprocessing
import time

from arguments import PretokenizationArguments
from datasets import load_dataset

from transformers import AutoTokenizer, HfArgumentParser


def tokenize(example):
    output = {}
    output["input_ids"] = tokenizer(example["content"], truncation=False)["input_ids"]
    output["ratio_char_token"] = len(example["content"]) / len(output["input_ids"])
    return output


parser = HfArgumentParser(PretokenizationArguments)
args = parser.parse_args()
if args.num_workers is None:
    args.num_workers = multiprocessing.cpu_count()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

t_start = time.time()
ds = load_dataset(args.dataset_name, split="train")
print(f"Dataset loaded in {time.time() - t_start:.2f}s")

t_start = time.time()
ds = ds.map(
    tokenize,
    num_proc=args.num_workers,
    remove_columns=[
        "repo_name",
        "path",
        "copies",
        "size",
        "content",
        "license",
        "hash",
        "line_mean",
        "line_max",
        "alpha_frac",
        "autogenerated",
    ],
)
print(f"Dataset tokenized in {time.time() - t_start:.2f}s")

t_start = time.time()
ds.push_to_hub(args.tokenized_data_repo)
print(f"Data pushed to the hub in {time.time() - t_start:.2f}s")
