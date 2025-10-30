import glob
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def _list_images(folder):
    # 여러 확장자 허용
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for pat in exts:
        files += glob.glob(os.path.join(folder, pat))
    return sorted(files)

class ImageDataset(Dataset):
    """
    CycleGAN 데이터셋 로더
    root: data/<dataset_name>  (trainA, trainB, testA, testB 구조)
    mode: 'train' 또는 'test'
    """
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        a_dir = os.path.join(root, f"{mode}A")
        b_dir = os.path.join(root, f"{mode}B")

        self.files_A = _list_images(a_dir)
        self.files_B = _list_images(b_dir)

        # 디버그: 실제 로딩된 개수와 경로 출력
        print(f"[dataset] root={root}")
        print(f"[dataset] {mode}A: {a_dir} -> {len(self.files_A)} files")
        print(f"[dataset] {mode}B: {b_dir} -> {len(self.files_B)} files")

        if len(self.files_A) == 0 or len(self.files_B) == 0:
            # 경로/확장자 문제를 빨리 알리기 위해 명시적 에러
            raise RuntimeError(
                f"[dataset] Empty split: {mode}A={len(self.files_A)}, {mode}B={len(self.files_B)}.\n"
                f"Check path and image extensions in {a_dir} / {b_dir}"
            )

    def __getitem__(self, index):
        # A
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")

        # B (unaligned면 랜덤, 아니면 동일 인덱스)
        if self.unaligned:
            img_B = Image.open(
                self.files_B[np.random.randint(0, len(self.files_B))]
            ).convert("RGB")
        else:
            img_B = Image.open(self.files_B[index % len(self.files_B)]).convert("RGB")

        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        # DataLoader가 정수를 기대하므로 int로 보장
        return int(max(len(self.files_A), len(self.files_B)))
