"""
Main training script for simplified ablation study.
Supports S0 (AASIST), S2 (SSL), S3 (SSL+AFSS), S4 (SSL+AFSS+Codec).

Simplified from AASIST (NAVER Corp.) — no TTA, no adversarial, no multi-task.
Allows CPU for smoke testing.
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import (SpoofDataset_train, SpoofDataset_devNeval,
                        genSpoof_list, gen_universal_spoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """Main function: trains, validates, and evaluates."""
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config.get("track", "LA")

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])

    # Protocol paths
    prefix_2019 = "ASVspoof2019.{}".format(track)
    dev_trial_path = (database_path / config.get("dev_list", 
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(track, prefix_2019)))
    eval_trial_path = (database_path / config.get("eval_list", 
                      "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(track, prefix_2019)))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config.get("eval_output",
                                              "eval_scores.txt")
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device — allow CPU for smoke tests
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        print("WARNING: Running on CPU. This is fine for smoke tests only.")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        print("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    # Optional SWA
    try:
        from torchcontrib.optim import SWA
        optimizer_swa = SWA(optimizer)
        use_swa = True
    except ImportError:
        print("torchcontrib not found, SWA disabled")
        optimizer_swa = None
        use_swa = False

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Optional TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(model_tag)
    except ImportError:
        writer = None
        print("TensorBoard not found, logging disabled")

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path / "dev_score.txt", dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path / "dev_score.txt",
            asv_score_file=database_path / config["asv_score_path"],
            output_file=metric_path / "dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
            running_loss, dev_eer, dev_tdcf))

        if writer:
            writer.add_scalar("loss", running_loss, epoch)
            writer.add_scalar("dev_eer", dev_eer, epoch)
            writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model found at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(
                           epoch, dev_eer))

            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            if use_swa:
                print("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1

        if writer:
            writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
            writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if use_swa and n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config["asv_score_path"],
        output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()

    torch.save(model.state_dict(), model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(), model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    nb_trainable = sum([p.view(-1).size()[0] for p in model.parameters()
                        if p.requires_grad])
    print("Total params: {}, Trainable: {}".format(nb_params, nb_trainable))
    return model


def get_loader(database_path, seed, config):
    """Make PyTorch DataLoaders for train / dev / eval"""
    track = config.get("track", "LA")
    prefix_2019 = "ASVspoof2019.{}".format(track)
    audio_ext = config.get("audio_ext", ".flac")
    audio_subdir = config.get("audio_subdir", "flac")

    # Augmentation flags
    codec_augment = str_to_bool(config.get("codec_augment", "False"))
    codec_config = config.get("codec_config", None)
    afss_augment = str_to_bool(config.get("afss_augment", "False"))
    afss_config = config.get("afss_config", None)

    trn_database_path = Path(database_path) / config.get("trn_dir", f"ASVspoof2019_{track}_train/")
    dev_database_path = Path(database_path) / config.get("dev_dir", f"ASVspoof2019_{track}_dev/")
    eval_database_path = Path(database_path) / config.get("eval_dir", f"ASVspoof2019_{track}_eval/")

    trn_list_path = Path(database_path) / config.get("trn_list", 
                     f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.train.trn.txt")
    dev_trial_path = Path(database_path) / config.get("dev_list", 
                      f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.dev.trl.txt")
    eval_trial_path = Path(database_path) / config.get("eval_list", 
                       f"ASVspoof2019_{track}_cm_protocols/{prefix_2019}.cm.eval.trl.txt")

    use_universal = str_to_bool(config.get("universal_protocol", "False"))
    fmt_config = config.get("protocol_format", {})

    if use_universal:
        d_label_trn, file_train = gen_universal_spoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False, fmt_config=fmt_config)
    else:
        d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path, is_train=True, is_eval=False)
        
    print("no. training files:", len(file_train))

    train_set = SpoofDataset_train(
        list_IDs=file_train, labels=d_label_trn,
        base_dir=trn_database_path, audio_ext=audio_ext,
        audio_subdir=audio_subdir,
        codec_augment=codec_augment, codec_config=codec_config,
        afss_augment=afss_augment, afss_config=afss_config)

    if use_universal:
        _, file_dev = gen_universal_spoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False, fmt_config=fmt_config)
    else:
        _, file_dev = genSpoof_list(dir_meta=dev_trial_path, is_train=False, is_eval=False)
        
    print("no. validation files:", len(file_dev))
    dev_set = SpoofDataset_devNeval(
        list_IDs=file_dev, base_dir=dev_database_path,
        audio_ext=audio_ext, audio_subdir=audio_subdir)

    if use_universal:
        file_eval = gen_universal_spoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True, fmt_config=fmt_config)
    else:
        file_eval = genSpoof_list(dir_meta=eval_trial_path, is_train=False, is_eval=True)
        
    eval_set = SpoofDataset_devNeval(
        list_IDs=file_eval, base_dir=eval_database_path,
        audio_ext=audio_ext, audio_subdir=audio_subdir)

    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set, batch_size=config["batch_size"],
                            shuffle=True, drop_last=True, pin_memory=True,
                            worker_init_fn=seed_worker, generator=gen)
    dev_loader = DataLoader(dev_set, batch_size=config["batch_size"],
                            shuffle=False, drop_last=False, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=config["batch_size"],
                             shuffle=False, drop_last=False, pin_memory=True)
    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(data_loader, model, device, save_path,
                            trial_path):
    """Perform evaluation and save scores to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            trl = trl.strip()
            # Try parsing as ASVspoof format first (5 columns, space separated)
            parts = trl.split(' ')
            if len(parts) == 5:
                _, utt_id, _, src, key = parts
            else:
                # Fallback for Universal/ThaiSpoof format (comma separated)
                parts = trl.split(',')
                utt_id = parts[0]
                key = parts[1] if len(parts) > 1 else "unknown"
                src = "unknown"
            
            # Remove any directory paths from the filename for the assert
            if '/' in utt_id:
                utt_id = utt_id.split('/')[-1]
            if '/' in fn:
                fn = fn.split('/')[-1]
                
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(trn_loader, model, optim, device, scheduler, config):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        _, batch_out = model(batch_x,
                             Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error")

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spoof detection ablation")
    parser.add_argument("--config", dest="config", type=str, required=True,
                        help="configuration file")
    parser.add_argument("--output_dir", dest="output_dir", type=str,
                        default="./exp_result",
                        help="output directory for results")
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="eval mode")
    parser.add_argument("--comment", type=str, default=None,
                        help="Comment to describe the saved model")
    main(parser.parse_args())
