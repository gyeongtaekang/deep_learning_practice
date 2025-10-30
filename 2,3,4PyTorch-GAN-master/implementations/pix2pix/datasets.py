# datasets.py
# Facades 다운로드/검증 + ImageDataset
import os, glob, tarfile, argparse, urllib.request, shutil, random, numpy as np
from typing import List
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 여러 미러를 순차 시도
MIRRORS = [
    "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz",  # 미러1(권장)
    "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz",  # 구버전(가끔 404)
]

def _count_images(dir_path: str) -> int:
    exts = (".jpg",".jpeg",".png",".bmp")
    if not os.path.isdir(dir_path): return 0
    return sum(1 for f in os.listdir(dir_path) if f.lower().endswith(exts))

def _try_download(url: str, out_path: str) -> bool:
    try:
        print(f"[datasets] downloading: {url}")
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception as e:
        print(f"[datasets] failed: {e}")
        return False

def _download_and_extract(dataroot: str, local_tgz: str | None = None) -> str:
    os.makedirs(dataroot, exist_ok=True)
    tgz_path = os.path.join(dataroot, "facades.tar.gz") if local_tgz is None else local_tgz
    if local_tgz is None:
        ok = False
        for u in MIRRORS:
            if _try_download(u, tgz_path):
                ok = True
                break
        if not ok:
            raise RuntimeError(
                "[datasets] 모든 미러에서 다운로드 실패. 수동으로 .tar.gz를 내려받아 "
                f"'{tgz_path}'(또는 --local_tgz 경로)로 놓고 다시 --ensure 하세요."
            )
    print("[datasets] extracting ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=dataroot)
    if local_tgz is None and os.path.exists(tgz_path):
        os.remove(tgz_path)
    extracted = os.path.join(dataroot, "facades")
    print(f"[datasets] extracted to: {extracted}")
    return extracted

def _fix_structure(dataroot: str) -> str:
    tr, te = os.path.join(dataroot,"train"), os.path.join(dataroot,"test")
    if _count_images(tr)>0 and _count_images(te)>0: return dataroot
    nested = os.path.join(dataroot,"facades")
    tr2, te2 = os.path.join(nested,"train"), os.path.join(nested,"test")
    if _count_images(tr2)>0 and _count_images(te2)>0: return nested
    nested2 = os.path.join(dataroot,"facades","facades")
    tr3, te3 = os.path.join(nested2,"train"), os.path.join(nested2,"test")
    if _count_images(tr3)>0 and _count_images(te3)>0:
        target = os.path.join(dataroot,"facades")
        print(f"[datasets] fixing nested structure: {nested2} -> {target}")
        for sub in ["train","test","val"]:
            src, dst = os.path.join(nested2,sub), os.path.join(target,sub)
            if os.path.isdir(dst): shutil.rmtree(dst)
            shutil.move(src, dst)
        shutil.rmtree(nested2)
        return target
    return dataroot

def ensure_facades(dataroot: str, local_tgz: str | None = None) -> str:
    resolved = _fix_structure(dataroot)
    if _count_images(os.path.join(resolved,"train"))>0 and _count_images(os.path.join(resolved,"test"))>0:
        return resolved
    _download_and_extract(dataroot, local_tgz=local_tgz)
    resolved = _fix_structure(dataroot)
    if _count_images(os.path.join(resolved,"train"))==0 or _count_images(os.path.join(resolved,"test"))==0:
        raise RuntimeError("[datasets] images not found after download.")
    return resolved

# ------------------ Dataset (그대로) ------------------
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A); img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}
    def __len__(self): return len(self.files)

# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default="./facades")
    ap.add_argument("--ensure", action="store_true", help="데이터 보장(없으면 다운로드)")
    ap.add_argument("--local_tgz", type=str, default=None,
                    help="수동으로 받은 facades.tar.gz 경로(옵션)")
    args = ap.parse_args()
    if args.ensure:
        resolved = ensure_facades(args.dataroot, local_tgz=args.local_tgz)
    else:
        resolved = _fix_structure(args.dataroot)
    tr, te = os.path.join(resolved,"train"), os.path.join(resolved,"test")
    print("[datasets] resolved:", resolved)
    print("[datasets] #train:", _count_images(tr), "| #test:", _count_images(te))
    if _count_images(tr)==0 or _count_images(te)==0:
        print("[datasets] 비어 있습니다. --ensure 또는 --local_tgz로 실행하세요.")
if __name__ == "__main__":
    main()
