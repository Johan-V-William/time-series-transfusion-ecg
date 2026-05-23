"""
preprocess_generated_ecg.py
Xử lý tín hiệu ECG dài (từ generative model) -> các beat kích thước cố định.
Hỗ trợ tự động phát hiện R-peak, gán nhãn (từ file hoặc dùng model dự đoán).
Kết quả: file .pt để kiểm thử với ECGTransForm.
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks
import wfdb  # nếu có annotation chuẩn
from sklearn.preprocessing import StandardScaler
import os

# ========= CẤU HÌNH ==========
FS = 360                     # Giả sử generative model sinh ở 360 Hz
BEAT_LEN = 186
LEFT = 90
RIGHT = 95
LOW_CUT = 0.5
HIGH_CUT = 40
DIST_BETWEEN_PEAKS = int(0.4 * FS)  # ít nhất 0.4s giữa 2 R-peak

def bandpass_filter(signal, fs=FS, lowcut=LOW_CUT, highcut=HIGH_CUT):
    nyquist = 0.5 * fs
    b, a = butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(signal, fs=FS):
    """
    Tự động phát hiện R-peak bằng phương pháp đơn giản: tìm đỉnh trên tín hiệu đã lọc.
    Có thể thay bằng Pan-Tompkins hoặc neurokit2 nếu cần chính xác hơn.
    """
    # Chuẩn hóa tín hiệu
    sig = (signal - np.mean(signal)) / np.std(signal)
    # Tìm đỉnh dương với độ cao ngưỡng
    peaks, properties = find_peaks(sig, height=0.5, distance=DIST_BETWEEN_PEAKS)
    return peaks

def extract_beats_from_signal(signal, r_peaks, left=LEFT, right=RIGHT):
    """Cắt beat xung quanh mỗi R-peak"""
    beats = []
    valid_indices = []
    for i, r in enumerate(r_peaks):
        start = r - left
        end = r + right
        if start >= 0 and end < len(signal):
            beat = signal[start:end]
            if len(beat) == BEAT_LEN:
                beats.append(beat)
                valid_indices.append(i)
    return np.array(beats), valid_indices

def assign_labels_with_model(beats, model, device, class_map=None):
    """
    Dùng mô hình ECGTransForm đã train để dự đoán nhãn cho từng beat.
    model: instance của ecgTransForm đã load checkpoint.
    """
    model.eval()
    model.to(device)
    batch_size = 128
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(beats), batch_size):
            batch = torch.from_numpy(beats[i:i+batch_size]).float().unsqueeze(1).to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

def load_pretrained_model(checkpoint_path, dataset_config, hparams, device):
    """Load model từ checkpoint của ECGTransForm"""
    from models import ecgTransForm
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ecgTransForm(configs=dataset_config, hparams=hparams)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_file', required=True, help='File .npy hoặc .txt chứa tín hiệu ECG (1D)')
    parser.add_argument('--annotation_file', default=None, help='File .atr hoặc .csv chứa nhãn beat (tùy chọn)')
    parser.add_argument('--out_pt', default='generated_test.pt', help='Đầu ra .pt')
    parser.add_argument('--use_model_for_label', action='store_true', help='Dùng mô hình đã train để gán nhãn nếu không có annotation')
    parser.add_argument('--checkpoint', default='experiments_logs/.../checkpoint_best.pt', help='Checkpoint của ECGTransForm (nếu dùng --use_model_for_label)')
    parser.add_argument('--dataset_type', default='mit', choices=['mit', 'ptb'], help='Loại dataset để load config')
    args = parser.parse_args()

    # 1. Đọc tín hiệu
    if args.signal_file.endswith('.npy'):
        signal = np.load(args.signal_file)
    else:
        signal = np.loadtxt(args.signal_file)
    if signal.ndim > 1:
        signal = signal[:, 0]  # lấy kênh đầu tiên
    print(f"Signal length: {len(signal)} samples")

    # 2. Lọc nhiễu
    filtered_signal = bandpass_filter(signal)

    # 3. Phát hiện R-peak
    r_peaks = detect_r_peaks(filtered_signal)
    print(f"Detected {len(r_peaks)} R-peaks")

    # 4. Cắt beat
    beats, valid_idx = extract_beats_from_signal(filtered_signal, r_peaks)
    print(f"Extracted {len(beats)} valid beats")

    # 5. Gán nhãn
    labels = None
    if args.annotation_file is not None:
        # Đọc annotation từ file (giả sử định dạng: mỗi dòng "sample_index label")
        ann = np.loadtxt(args.annotation_file)
        ann_samples = ann[:, 0].astype(int)
        ann_labels = ann[:, 1].astype(int)
        # Map R-peak đã phát hiện với annotation gần nhất
        labels = []
        for r in r_peaks[valid_idx]:
            # Tìm annotation sample gần nhất (trong khoảng 10 mẫu)
            diffs = np.abs(ann_samples - r)
            if np.min(diffs) <= 10:
                idx = np.argmin(diffs)
                labels.append(ann_labels[idx])
            else:
                labels.append(-1)  # không tìm thấy
        labels = np.array(labels)
        # Bỏ những beat không có nhãn (nếu muốn)
        # keep = labels != -1
        # beats = beats[keep]
        # labels = labels[keep]
        print(f"Assigned labels for {np.sum(labels!=-1)} beats from annotation")

    elif args.use_model_for_label:
        # Load model đã train để dự đoán
        from configs.data_configs import get_dataset_class
        from configs.hparams import get_hparams_class
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset_config = get_dataset_class(args.dataset_type)()
        hparams_class = get_hparams_class("supervised")
        hparams = hparams_class.train_params
        model = load_pretrained_model(args.checkpoint, dataset_config, hparams, device)
        labels = assign_labels_with_model(beats, model, device)
        print(f"Predicted labels using model, shape {labels.shape}")
    else:
        # Không có nhãn: gán label mặc định 0 (cần xử lý sau)
        labels = np.zeros(len(beats), dtype=np.int64)
        print("Warning: No labels assigned, defaulting to 0")

    # 6. Đảm bảo labels là int64
    labels = labels.astype(np.int64)

    # 7. Lưu thành .pt
    dataset = {
        'samples': beats.astype(np.float32),
        'labels': labels
    }
    torch.save(dataset, args.out_pt)
    print(f"Saved to {args.out_pt}, beats shape {beats.shape}, labels shape {labels.shape}")

if __name__ == '__main__':
    main()
