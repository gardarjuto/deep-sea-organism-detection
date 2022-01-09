import os
import logging
import torch
import torch.cuda
from torch.distributed import init_process_group, get_rank


def initialise_distributed(args):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logging.info("Not using distributed mode")
        args.distributed = False
        return
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


def initialise_logging(args):
    level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=level, format='%(asctime)s %(message)s',
                            datefmt='%I:%M:%S %p')
    else:
        logging.basicConfig(level=level, format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')


def collate_fn(batch):
    return tuple(zip(*batch))


def save_state(checkpoint, output_dir, epoch, distributed):
    if distributed and get_rank() != 0:
        return
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.file'))
    torch.save(checkpoint, os.path.join(output_dir, f'model_{epoch}.file'))
