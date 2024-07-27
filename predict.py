import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("model_path_or_url", type=str)
    parser.add_argument("kigos", type=str)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--n_samples", type=int, default=3)
    return parser.parse_args()


model_path = os.path.join(
    os.path.dirname(__file__), 
    "outputs/checkpoint-12450",
)


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-xsmall", use_fast=False)
    tokenizer.do_lower_case = True
    model = AutoModelForCausalLM.from_pretrained(args.model_path_or_url)

    text = tokenizer.bos_token + args.kigos + tokenizer.sep_token
    input = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input, do_sample=True, max_length=args.max_length, num_return_sequences=args.n_samples)
    for i, out in enumerate(outputs):
        sentence = tokenizer.decode(out.tolist())
        print(i+1, sentence, "\n")