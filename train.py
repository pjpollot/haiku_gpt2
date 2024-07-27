import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from argparse import ArgumentParser, Namespace


this_dir = os.path.dirname(__file__)


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path_or_url", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default=os.path.join(this_dir, "outputs"))
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("-n", "--num_epochs", type=int, default=10)
    return parser.parse_args()


repo_id = "rinna/japanese-gpt2-xsmall"


if __name__ == "__main__":
    args = parse_arguments()

    dataset = load_dataset("pjpollot/kigos_and_haikus")

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    tokenizer.do_lower_case = True

    model = AutoModelForCausalLM.from_pretrained(args.model_path_or_url or repo_id)

    def preprocess(examples: dict) -> dict:
        global n_errors
        kigos = examples.pop("kigos")
        haiku = examples.pop("haiku")
        # format: '<s>Kigos[SEP]Haiku</s>' + [PAD] until reaching max length
        text = tokenizer.bos_token + kigos + tokenizer.sep_token + haiku + tokenizer.eos_token
        results = tokenizer(text, max_length=args.max_length, padding="max_length", return_tensors="pt")
        results["labels"] = results["input_ids"].clone()
        return results

    dataset = dataset.map(preprocess)
    print("== Without filtering ==\n", dataset)
    dataset = dataset.filter(lambda examples: tokenizer.unk_token_id not in examples["input_ids"][0]) # remove the rows where the kanjis are unknown
    print("== After filtering ==\n", dataset)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        #fp16=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        optim="adamw_torch",
        save_safetensors=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=os.path.join(this_dir, "logs"),
    )

    trainer = Trainer(model, training_args, train_dataset=dataset["train"], eval_dataset=dataset["test"])
    trainer.train()

    trainer.save_model()

    results = trainer.evaluate()
