import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-l", action="store_true", help="Run latency mode")
parser.add_argument("-b", action="store_true", help="Run burst mode")
parser.add_argument("-r", action="store_true", help="Run random mode")
parser.add_argument("-c", action="store_true", help="Run mops-controlled mode")
parser.add_argument("-x", "--cxi", action="store_true", help="Run CXI post/completion mode")
parser.add_argument("--cxi-bytes", type=int, default=8, help="Bytes per CXI write in CXI mode")
parser.add_argument(
    "--cxi-max-outstanding",
    type=int,
    default=1024,
    help="Maximum outstanding CXI writes per proxy in CXI mode",
)
args = parser.parse_args()

if args.l:
    cmd = ["./benchmark", "-l"]
elif args.b:
    cmd = ["./benchmark", "-b"]
elif args.r:
    cmd = ["./benchmark", "-r"]
elif args.c:
    cmd = ["./benchmark", "-c"]
elif args.cxi:
    cmd = [
        "./benchmark",
        "-x",
        "--cxi-bytes",
        str(args.cxi_bytes),
        "--cxi-max-outstanding",
        str(args.cxi_max_outstanding),
    ]
else:
    cmd = ["./benchmark"]

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(f"Running: {local_rank}/{world_size} {cmd}")

if local_rank == 0:
    subprocess.run(cmd, check=True)
else:
    with open("/dev/null", "w") as devnull:
        subprocess.run(cmd, check=True, stdout=devnull, stderr=devnull)
