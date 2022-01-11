import os
import logging
import sys
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
    torch.distributed.barrier(device_ids=args.gpu)


def initialise_logging(args):
    if not is_master_process():
        logging.disable()
        return
    level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=level, format='[%(asctime)s] %(message)s',
                            datefmt='%I:%M:%S %p')
    else:
        logging.basicConfig(stream=sys.stdout, level=level, format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')


def collate_fn(batch):
    return tuple(zip(*batch))


def is_master_process():
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return False
    return True


def save_state(checkpoint, output_dir, epoch):
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.file'))
    torch.save(checkpoint, os.path.join(output_dir, f'model_{epoch}.file'))


def tensor_encode_id(img_id):
    """
    Encodes a FathomNet image id like '00a6db92-5277-4772-b019-5b89c6af57c3' as a tensor
    of shape torch.Size([4]) of four integers in the range [0, 2^32-1].
    """
    hex_str = img_id.replace('-', '')
    length = len(hex_str) // 4
    img_id_enc = tuple(int(hex_str[i * length: (i + 1) * length], 16) for i in range(4))
    return torch.tensor(img_id_enc)


def tensor_decode_id(img_id_enc):
    ints = img_id_enc.tolist()
    img_id = ''.join([hex(part)[2:].zfill(8) for part in ints])
    for ind in [8, 13, 18, 23]:
        img_id = img_id[:ind] + '-' + img_id[ind:]
    return img_id
