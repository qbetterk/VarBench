import random, argparse, time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt-4-turbo", #"gpt-4-1106-preview", # "gpt-3.5-turbo", # "gpt-4", "gpt-3.5-turbo-1106"
        help="The name of the model for generation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="The name of function to call",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Claim dir for saving files",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="",
        help="Claim filename for saving files",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Path to data file",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./prompt/prompt.txt",
        help="Claim path to prompt file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for decoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum decoding length",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Maximum decoding length",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Penalty for token frequency",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Penalty for token presence",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    args = parser.parse_args()
    return args
