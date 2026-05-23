
import torch
import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, exp_log_dir, seed_id):
    log_dir = os.path.join(exp_log_dir, "_seed_" + str(seed_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug("=" * 45)
    logger.debug(f'Seed: {seed_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(exp_log_dir, model, dataset, dataset_configs, hparams, status):
    save_dict = {
        "dataset": dataset,
        "configs": dataset_configs.__dict__,
        "hparams": dict(hparams),
        "model": model.state_dict()
    }
    # save classification report
    save_path = os.path.join(exp_log_dir, f"checkpoint_{status}.pt")

    torch.save(save_dict, save_path)


def _calc_metrics(pred_labels, true_labels, classes_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=classes_names, digits=6, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def _save_metrics(pred_labels, true_labels, log_dir, status):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = f"classification_report_{status}.xlsx"
    report_Save_path = os.path.join(log_dir, file_name)
    df.to_excel(report_Save_path)


import collections


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models.py", os.path.join(destination_dir, f"models.py"))
    copy(f"configs/data_configs.py", os.path.join(destination_dir, f"data_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"trainer.py", os.path.join(destination_dir, f"trainer.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))


def _plot_umap(model, data_loader, device, save_dir):
    import umap
    import umap.plot
    from matplotlib.colors import ListedColormap
    classes_names = ['N','S','V','F','Q']
    
    font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 17}
    plt.rc('font', **font)
    
    with torch.no_grad():
        # Source flow
        data = data_loader.dataset.x_data.float().to(device)
        labels = data_loader.dataset.y_data.view((-1)).long()
        out = model[0](data)
        features = model[1](out)


    if not os.path.exists(os.path.join(save_dir, "umap_plots")):
        os.mkdir(os.path.join(save_dir, "umap_plots"))
        
    #cmaps = plt.get_cmap('jet')
    model_reducer = umap.UMAP() #n_neighbors=3, min_dist=0.3, metric='correlation', random_state=42)
    embedding = model_reducer.fit_transform(features.detach().cpu().numpy())
    
    # Normalize the labels to [0, 1] for colormap
    norm_labels = labels / 4.0
    

    # Create a new colormap by extracting the first 5 colors from "Paired"
    paired = plt.cm.get_cmap('Paired', 12)  # 12 distinct colors
    new_colors = [paired(0), paired(1), paired(2), paired(4), paired(6)]  # Skip every second color, but take both from the first pair
    new_cmap = ListedColormap(new_colors)

    print("Plotting UMAP ...")
    plt.figure(figsize=(16, 10))
    # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels,  s=10, cmap='Spectral')
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=norm_labels, cmap=new_cmap, s=15)

    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, classes_names,  title="Classes")
    file_name = "umap_.png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()


def calculate_rr_intervals(annotation_samples, sampling_frequency):
    """Calculates R-R intervals in seconds."""
    rr_samples = np.diff(annotation_samples) # Difference between consecutive R-peak samples
    rr_intervals_sec = rr_samples / sampling_frequency
    return rr_intervals_sec


def calculate_heart_rate(rr_intervals_sec):
    """Calculates instantaneous heart rate in BPM."""
    # Heart rate = 60 / RR-interval (in seconds)
    heart_rates = 60 / rr_intervals_sec
    return heart_rates

def calculate_hrv_sdnn(rr_intervals_sec):
    """Calculates SDNN (Standard Deviation of NN intervals), a common HRV metric."""
    if len(rr_intervals_sec) < 2:
        return np.nan # Not enough intervals to calculate std dev
    return np.std(rr_intervals_sec)

def validate_beat(
    segment:  np.ndarray,
    min_len:  int = 50,
    max_len:  int = 512,
) -> bool:
    \"\"\"Sanity check before feeding to CNN.\"\"\"
    if len(segment) < min_len or len(segment) > max_len:
        return False
    if np.ptp(segment) < 1e-6:   # flat signal
        return False
    return True


def snr_estimate(segment: np.ndarray) -> float:
    \"\"\"\n    Rough SNR estimate: ratio of peak-to-peak to noise floor.
    Noise estimated as std of signal after removing trend.
    \"\"\"
    trend = np.linspace(segment[0], segment[-1], len(segment))
    residual = segment - trend
    signal_power = np.ptp(segment)
    noise_power  = residual.std()
    return signal_power / (noise_power + 1e-8)

def plot_segmentation(signal, beats, title="ECG Segmentation", fs=360):
    \"\"\"\n    Matplotlib visualization matching Figure 10 from paper:
    - Top: full signal with window positions + segmented beat boundaries
    - Bottom: zoomed individual beat segments (Method I and II)

    Requires matplotlib.
    \"\"\"
    # try:
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    # except ImportError:
    #     print("pip install matplotlib")
    #     return

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    time = np.arange(len(signal)) / fs

    # Top: full signal + boundaries
    ax = axes[0]
    ax.plot(time, signal, "royalblue", lw=0.8, label="ECG")
    for beat in beats:
        # Note: Assuming 'beats' objects have 'bl_I', 'br_I', 'cp_abs' attributes
        if hasattr(beat, 'bl_I') and hasattr(beat, 'br_I') and beat.bl_I is not None and beat.br_I is not None:
            ax.axvspan(beat.bl_I/fs, beat.br_I/fs, alpha=0.15, color="orange", label="Method I")
        if hasattr(beat, 'cp_abs'):
            ax.axvline(beat.cp_abs/fs, color="red", lw=0.5, alpha=0.5)
    ax.set_title(title); ax.legend(loc="upper right", fontsize=8)

    # Middle: Method I segments (last 2 beats)
    ax = axes[1]; ax.set_title("Method I segments")
    for beat in beats[-2:]:
        if hasattr(beat, 'segment_I'):
            seg = beat.segment_I(signal)
            if seg is not None:
                ax.plot(seg, lw=1.2)

    # Bottom: Method II segments (last 2 beats)
    ax = axes[2]; ax.set_title("Method II segments")
    for beat in beats[-2:]:
        if hasattr(beat, 'segment_II'):
            seg = beat.segment_II(signal)
            if seg is not None:
                ax.plot(seg, lw=1.2)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.show()

