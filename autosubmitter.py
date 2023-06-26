#!/usr/bin/env python3
"""Autosubmits sbatch jobs to queue."""
import argparse
import os
import glob
from time import sleep

SLEEP = 5

# def main(args):
#     script_dir  = os.path.join(args.basedir, "batch_scripts", args.name)
#     os.chdir(args.basedir)
#     for name in glob.glob(os.path.join(script_dir, "*.sh")):
#         cmd  = 'sbatch ' + name
#         print('Running: ' + cmd)

#         os.system(cmd)
#         sleep(SLEEP)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Autosubmits sbatch jobs to queue.")
#     parser.add_argument("-b", "--basedir", default=os.getcwd(), help="base directory")
#     parser.add_argument("-n", "--name", help="model name")
#     args = parser.parse_args()
#     main(args)


# def main(args):
#     script_dir = os.path.join(args.basedir, "batch_scripts", args.name)
#     os.chdir(args.basedir)
#     for name in glob.glob(os.path.join(script_dir, "*.sh")):
#         cmd = "sbatch " + name
#         print("Running: " + cmd)

#         os.system(cmd)
#         sleep(SLEEP)


if __name__ == "__main__":
    for i in range(1, 8):
        if i == 3:
            continue
        cmd = f"sbatch batch_scripts/CORAL/try{i}-sbatch-celldart_rv.sh"
        print("Running: " + cmd)
        os.system(cmd)
        sleep(SLEEP)
