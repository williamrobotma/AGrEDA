#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import argparse
import datetime
import logging

from src.da_models.model_utils.utils import get_metric_ctp
from src.da_utils.evaluator import Evaluator

# self.args_dict['modelname'] = self.args_dict['modelname
# self.args_dict['milisi'] = self.args_dict['milisi


metric_ctp = get_metric_ctp("cos")

# device = get_torch_device(self.args_dict['cuda)


def main(args):
    evaluator = Evaluator(vars(args), metric_ctp)
    evaluator.evaluate_embeddings()
    evaluator.eval_spots()
    evaluator.eval_sc()

    evaluator.produce_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates.")
    parser.add_argument("--pretraining", "-p", action="store_true", help="force pretraining")
    parser.add_argument("--modelname", "-n", type=str, default="ADDA", help="model name")
    parser.add_argument("--milisi", "-m", action="store_false", help="no milisi")
    parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
    parser.add_argument(
        "--njobs", type=int, default=1, help="Number of jobs to use for parallel processing."
    )
    parser.add_argument("--cuda", "-c", default=None, help="GPU index to use")
    parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary results directory")
    parser.add_argument("--test", "-t", action="store_true", help="test mode")
    args = parser.parse_args()

    script_start_time = datetime.datetime.now(datetime.timezone.utc)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )
    main(args)
    print("Script run time:", datetime.datetime.now(datetime.timezone.utc) - script_start_time)
