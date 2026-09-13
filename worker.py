import os, sys
os.environ["QUANT_WORKER"]="1"
from app import worker_main
if __name__ == "__main__":
    if len(sys.argv)!=2: raise SystemExit("worker.py requires job id")
    worker_main(sys.argv[1])
