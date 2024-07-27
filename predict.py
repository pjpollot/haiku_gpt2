import os

from argparse import ArgumentParser, Namespace

from pipeline import KigosHaikuPipeline


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

    pipe = KigosHaikuPipeline(args.model_path_or_url)

    outputs = pipe.sample_haikus(args.kigos, n_samples=args.n_samples, max_length=args.max_length)
    for i, sentence in enumerate(outputs):
        print(i+1, sentence, "\n")