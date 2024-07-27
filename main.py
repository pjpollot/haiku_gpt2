import os

from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = os.path.join(
    os.path.dirname(__file__), 
    "outputs/checkpoint-12450",
)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-xsmall", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    text = "<s>桜[SEP]"
    input = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input, do_sample=True, max_length=50, num_return_sequences=3)
    print(tokenizer.batch_decode(outputs))