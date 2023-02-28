import argparse
import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter, accuracy
from torch.utils.data import Dataset  # For custom datasets
from tqdm import tqdm
from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer
from scheduler import build_scheduler
from utils import (
    create_logger,
    load_checkpoint,
    save_checkpoint,
    NativeScalerWithGradNormCount,
)


def parse_option():
    parser = argparse.ArgumentParser(
        "Vision model training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument(
        "--batch-size", type=int, help="batch size for single GPU"
    )
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Perform evaluation only"
    )
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
    ) = build_loader(config)

    model = build_model(config)
    logger.info(str(model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # n_flops =

    # can be simple
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = NativeScalerWithGradNormCount()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model, optimizer, lr_scheduler, logger
        )
        acc1, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_acc1, train_loss = train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            scaler,
        )
        logger.info(
            f" * Train Acc {train_acc1:.3f} Train Loss {train_loss:.3f}"
        )
        logger.info(
            f"Accuracy of the network on the {len(dataset_train)} train images: {train_acc1:.1f}%"
        )

        train_acc1, _ = validate(config, data_loader_train, model)
        val_acc1, val_loss = validate(config, data_loader_val, model)
        logger.info(
            f" * Train Acc {train_acc1:.3f} Test Acc {val_acc1:.3f} Test Loss {val_loss:.3f}"
        )
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {val_acc1:.1f}%"
        )

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(
                config,
                epoch,
                model,
                max_accuracy,
                optimizer,
                lr_scheduler,
                logger,
            )

        max_accuracy = max(max_accuracy, val_acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    scaler,
):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(tqdm(data_loader, leave=False)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        if config.MODEL.NAME == "revvit":
            outputs = model(samples)
        else:
            with torch.cuda.amp.autocast(enabled=config.ENABLE_AMP):
                outputs = model(samples)

        loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()
        # Scaler implicitly has backward() and then step() in it
        grad_norm = scaler(loss, optimizer, parameters=model.parameters())

        lr_scheduler.step_update(epoch * num_steps + idx)
        loss_scale_value = scaler.state_dict()["scale"]

        (acc1,) = accuracy(outputs, targets)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        (acc1,) = accuracy(output, target)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    logger.info(f" * Test Acc@1 {acc1_meter.avg:.3f}")
    return acc1_meter.avg, loss_meter.avg


if __name__ == "__main__":
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # Make output dir
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}"
    )

    path = os.path.join(config.OUTPUT, "config.yaml")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
