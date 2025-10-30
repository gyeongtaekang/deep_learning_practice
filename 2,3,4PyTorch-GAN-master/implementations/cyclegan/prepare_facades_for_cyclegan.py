# prepare_facades_for_cyclegan.py
# pix2pix용 facades(좌 A | 우 B)를 cyclegan용 비정렬(unpaired) 구조로 변환
import os, glob
from PIL import Image

# 원본 facades 위치(앞에서 받은 데이터): ..\pix2pix\facades\facades\{train,test}
SRC = r"..\pix2pix\facades\facades"
# CycleGAN이 읽는 위치: .\data\facades_cyclegan\{trainA,trainB,testA,testB}
DST = r".\data\facades_cyclegan"

def split_set(src_dir, dstA, dstB):
    os.makedirs(dstA, exist_ok=True)
    os.makedirs(dstB, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not files:
        raise SystemExit(f"[err] no images in {src_dir}. Check SRC path.")
    for p in files:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        A = img.crop((0, 0, w // 2, h))    # 왼쪽(A)
        B = img.crop((w // 2, 0, w, h))    # 오른쪽(B)
        base = os.path.splitext(os.path.basename(p))[0]
        A.save(os.path.join(dstA, base + ".jpg"))
        B.save(os.path.join(dstB, base + ".jpg"))

if __name__ == "__main__":
    train_src = os.path.join(SRC, "train")
    test_src  = os.path.join(SRC, "test")

    split_set(train_src, os.path.join(DST, "trainA"), os.path.join(DST, "trainB"))
    split_set(test_src,  os.path.join(DST, "testA"),  os.path.join(DST, "testB"))

    # 개수 출력
    for sub in ["trainA","trainB","testA","testB"]:
        d = os.path.join(DST, sub)
        n = len(glob.glob(os.path.join(d, "*.jpg")))
        print(f"[ok] {sub}: {n}")
