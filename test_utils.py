import os


def set_dist_debug():
    
    log_format = "%(asctime)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["TORCH_LOGS_FORMAT"] = log_format
    os.environ["TORCH_TRACE"] = './torch_trace'
    os.environ["TORCH_LOGS_OUT"] = './torch_log'
    os.environ["TORCH_LOGS"] = "all"
    
    check_torch_distributed_debug_level()

fh = None

def log_rank_file(rank, *msgs):
    """
    Print to a log file of the given rank

    This is useful for debugging hanging in sync processes. Here is a possible workflow:

    1. Enable the force debug in say partitioning and zero3 files
    2. Override the usual versions of print_rank_0 in those files with ::

        def print_rank_0(message, debug=False, force=False):
            rank = deepspeed.comm.get_rank()
            log_rank_file(rank, message)

    3. run the program
    4. fix up the expected differences, e.g. different cuda numbers ::

        perl -pi -e 's|cuda:1|cuda:0|' log_rank_*

    5. now diff and see where names and ids diverge - you will find where the gpus don't do the same
    work (e.g. when some layers get conditionally skipped on one gpu but not all)

        diff -u log_rank_0.txt log_rank_1.txt | less

    """
    global fh
    if fh is None:
        fh = open(f"log_rank_{rank}.txt", "w")
    for m in msgs:
        fh.write(f"{m}\n")
    fh.flush()

def check_torch_distributed_debug_level():
    import torch
    log_state = torch._logging._internal._get_log_state()
    print(f"{log_state=}")
    torch_logs = os.environ["TORCH_LOGS"]
    print(f"{torch_logs=}")
    print(f"{torch.distributed.get_debug_level()=}")
    