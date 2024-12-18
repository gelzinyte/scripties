from mace import tools
import torch
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model")
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run(args)

def run(args:argparse.Namespace) -> None:

    model = torch.load(f=args.model, map_location=args.device)
    num_params = tools.count_parameters(model)

    print(f"Number of parameters: {num_params}")

if __name__ == "__main__":
    main()
