import os
import logging
import torch
import torch.cuda
import torch.distributed as dist


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
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


def initialise_logging(args):
    if not is_master_process():
        logging.disable()
        return
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


def is_master_process():
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return False
    return True


def save_state(checkpoint, output_dir, epoch):
    if is_master_process():
        torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.file'))
        torch.save(checkpoint, os.path.join(output_dir, f'model_{epoch}.file'))
