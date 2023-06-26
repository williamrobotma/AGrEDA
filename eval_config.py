#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import argparse
import datetime
import logging

import torch
from joblib import effective_n_jobs

from src.da_models.model_utils.utils import get_metric_ctp
from src.da_utils.evaluator import Evaluator


metric_ctp = get_metric_ctp("cos")


def main(args):
    torch.set_num_threads(effective_n_jobs(int(args.njobs)))
    evaluator = Evaluator(vars(args), metric_ctp)
    evaluator.eval_spots()
    evaluator.evaluate_embeddings()
    evaluator.eval_sc()

    evaluator.produce_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates.")
    parser.add_argument("--pretraining", "-p", action="store_true", help="force pretraining")
    parser.add_argument("--modelname", "-n", type=str, default="ADDA", help="model name")
    parser.add_argument("--milisi", "-m", action="store_false", help="no milisi")
    parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
    parser.add_argument("--configs_dir", "-cdir", type=str, default="configs", help="config dir")
    parser.add_argument(
        "--njobs", type=int, default=1, help="Number of jobs to use for parallel processing."
    )
    parser.add_argument("--cuda", "-c", default=None, help="GPU index to use")
    parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary results directory")
    parser.add_argument("--test", "-t", action="store_true", help="test mode")
    parser.add_argument(
        "--early_stopping",
        "-e",
        action="store_true",
        help="evaluate early stopping. Default: False",
    )
    parser.add_argument(
        "--reverse_val",
        "-r",
        action="store_true",
        help="use best model through reverse validation. Will use provided"
        "config file to search across models, then use the one loaded. Default: False",
    )

    args = parser.parse_args()

    script_start_time = datetime.datetime.now(datetime.timezone.utc)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )
    main(args)
    print("Script run time:", datetime.datetime.now(datetime.timezone.utc) - script_start_time)
