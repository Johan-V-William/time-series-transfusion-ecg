import argparse
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_processed_data(file_path):
    """Load file .pt được sinh ra từ script tiền xử lý"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Khong tim thay file: {file_path}")
    
    data = torch.load(file_path)
    samples = data['samples']
    labels = data['labels']
    
    # Chuyển đổi về numpy để dễ xử lý chia tách bằng sklearn
    if isinstance(samples, torch.Tensor):
        samples = samples.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
        
    return samples, labels

def save_dataset(samples, labels, out_path):
    """Lưu dữ liệu theo đúng cấu trúc dictionary cho ECGTransForm"""
    dataset = {
        'samples': torch.from_numpy(samples).float(),
        'labels': torch.from_numpy(labels).long()
    }
    torch.save(dataset, out_path)
    print(f"Da luu: {out_path} | Samples shape: {samples.shape} | Labels shape: {labels.shape}")

def main():
    parser = argparse.ArgumentParser(description="Xay dung bo du lieu TSTR va Mixed cho ECGTransForm")
    parser.add_argument('--real_pt', required=True, help='Duong dan file real_data.pt da tien xu ly')
    parser.add_argument('--syn_pt', required=True, help='Duong dan file synthetic_data.pt da tien xu ly')
    parser.add_argument('--out_dir', default='./data_splits', help='Thu muc dau ra de luu cac tap train/test')
    parser.add_argument('--mode', required=True, choices=['syn_train_real_test', 'real_train_syn_test', 'mixed'],
                        help='Kich ban kiem thu: syn_train_real_test (Muc 2/3), real_train_syn_test, mixed (Muc 1)')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Ty le chia tap train trong che do mixed (vi du: 0.7 hoac 0.6)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed de dam bao tinh tai hien')
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Dang doc du lieu dau vao...")
    real_samples, real_labels = load_processed_data(args.real_pt)
    syn_samples, syn_labels = load_processed_data(args.syn_pt)
    
    print(f"Real data: {real_samples.shape[0]} chu ky")
    print(f"Synthetic data: {syn_samples.shape[0]} chu ky")
    
    train_samples, train_labels = None, None
    test_samples, test_labels = None, None
    
    # KICH BAN 1: Train on Synthetic, Test on Real (Substitution / Clinical Grade)
    if args.mode == 'syn_train_real_test':
        print("\n--- Che do: Train on Synthetic, Test on Real ---")
        train_samples, train_labels = syn_samples, syn_labels
        test_samples, test_labels = real_samples, real_labels
        
    # KICH BAN 2: Train on Real, Test on Synthetic (Kiem tra nguoc do bao phu cua du lieu sinh)
    elif args.mode == 'real_train_syn_test':
        print("\n--- Che do: Train on Real, Test on Synthetic ---")
        train_samples, train_labels = real_samples, real_labels
        test_samples, test_labels = syn_samples, syn_labels
        
    # KICH BAN 3: Mixed Data (Augmentation Grade)
    elif args.mode == 'mixed':
        print(f"\n--- Che do: Mixed Data (Split ratio: {args.split_ratio}/{1-args.split_ratio:.1f}) ---")
        
        # Gop ca 2 tap du lieu lai
        mixed_samples = np.concatenate([real_samples, syn_samples], axis=0)
        mixed_labels = np.concatenate([real_labels, syn_labels], axis=0)
        
        # Chia tach phan tang (Stratified Split) de giu nguyen ti le cac lop loan nhip
        train_samples, test_samples, train_labels, test_labels = train_test_split(
            mixed_samples, 
            mixed_labels, 
            train_size=args.split_ratio, 
            random_state=args.seed,
            stratify=mixed_labels
        )

    # Thong ke phan phoi cac lop truoc khi luu
    print("\nThong ke phan phoi nhan trong tap Train:")
    classes, counts = np.unique(train_labels, return_counts=True)
    for c, cnt in zip(classes, counts):
        print(f"  Class {c}: {cnt} samples")
        
    print("Thong ke phan phoi nhan trong tap Test:")
    classes_t, counts_t = np.unique(test_labels, return_counts=True)
    for c, cnt in zip(classes_t, counts_t):
        print(f"  Class {c}: {cnt} samples")

    # Xuat file dau ra
    print("\nDang tien hanh luu cac file dinh dang .pt...")
    save_dataset(train_samples, train_labels, os.path.join(args.out_dir, f'{args.mode}_train.pt'))
    save_dataset(test_samples, test_labels, os.path.join(args.out_dir, f'{args.mode}_test.pt'))
    print("Hoan thanh!")

if __name__ == '__main__':
    main()
