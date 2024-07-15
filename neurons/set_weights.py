import sys
from argparse import ArgumentParser

import bittensor as bt
import torch

from bettensor.validator.bettensor_validator import BettensorValidator


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--alpha", type=float, default=0.9, help="The alpha value for the validator."
    )

    parser.add_argument("--netuid", type=int, default=30, help="The chain subnet uid.")

    parser.add_argument(
        "--max_targets",
        type=int,
        default=128,
        help="Sets the value for the number of targets to query at once",
    )
    parser.add_argument(
        "--load_state",
        type=str,
        default="True",
        help="WARNING: Setting this value to False clears the old state.",
    )

    validator = BettensorValidator(parser=parser)
    if (
            not validator.apply_config(bt_classes=[bt.subtensor, bt.logging, bt.wallet])
            or not validator.initialize_neuron()
    ):
        bt.logging.error("Unable to initialize Validator. Exiting.")
        sys.exit()

    try:
        validator.metagraph = validator.sync_metagraph_lite(
            validator.metagraph, validator.subtensor
        )
        bt.logging.debug(f"Metagraph synced: {validator.metagraph}")
    except TimeoutError as e:
        bt.logging.error(f"Metagraph sync timed out: {e}")
        sys.exit()

    validator.check_hotkeys()

    # If there are more axons than scores, append the scores list and add new miners to the database
    if len(validator.metagraph.uids.tolist()) > len(validator.scores):
        bt.logging.info(
            f"Discovered new Axons, current scores: {validator.scores}"
        )
        validator.scores = torch.cat(
            (
                validator.scores,
                torch.zeros(
                    (
                            len(validator.metagraph.uids.tolist())
                            - len(validator.scores)
                    ),
                    dtype=torch.float32,
                ),
            )
        )
        bt.logging.info(f"Updated scores, new scores: {validator.scores}")

    validator.add_new_miners()

    validator.set_weights_hc()


if __name__ == "__main__":
    main()
