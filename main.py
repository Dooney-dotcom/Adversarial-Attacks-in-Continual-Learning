import sys
from taba.experiment import run_taba_experiment
from aacl.experiment import run_aacl_experiment

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "taba":
        run_taba_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "aacl":
        run_aacl_experiment()
    else:
        print("No valid argument provided.\nUse 'taba' to run the TABA experiment.\nUse 'aacl' to run the AACL experiment.")
