import argparse

def parse_args():
    """
    Parse arguments given to the script.
    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="Run distributed data-parallel training and log with wandb.")
    # Used for `distribution.launch`
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    parser.add_argument(
        "--log_all",
        action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        metavar="N",
        help="number of data samples in one batch",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="wandb project",
    )
    args = parser.parse_args()
    return args