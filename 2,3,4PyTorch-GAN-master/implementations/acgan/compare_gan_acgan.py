#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
compare_gan_acgan.py
- 1번(GAN)과 2번(ACGAN) 생성 이미지를 좌/우로 붙여 한 장의 비교 이미지를 생성
- 기본 동작: 각 폴더에서 '가장 최근 PNG'를 자동으로 찾아 사용
- 옵션:
    --gan-png   : GAN 쪽 PNG 직접 지정
    --acgan-png : ACGAN 쪽 PNG 직접 지정
    --out       : 출력 파일 경로 지정 (기본: C:\Users\AERO\Desktop\dip_pra\compare_gan_acgan.png)

사용 예)
    python compare_gan_acgan.py
    python compare_gan_acgan.py --gan-png "runs\mnist_gan\samples\samples_epoch_020.png"
    python compare_gan_acgan.py --acgan-png "PyTorch-GAN-master\images\acgan\mnist\199.png"
    python compare_gan_acgan.py --out "C:\Users\AERO\Desktop\dip_pra\outputs\my_compare.png"
"""

import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont

BASE = r"C:\Users\AERO\Desktop\dip_pra"

# 1번(GAN) 샘플 폴더
GAN_DIR = os.path.join(BASE, r"runs\mnist_gan\samples")
# 2번(ACGAN) 샘플 후보 폴더(레포 버전에 따라 다를 수 있어 두 곳 모두 탐색)
ACGAN_CANDIDATES = [
    os.path.join(BASE, r"PyTorch-GAN-master\images\acgan\mnist"),
    os.path.join(BASE, r"PyTorch-GAN-master\implementations\acgan\images"),
]

DEFAULT_OUT = os.path.join(BASE, "compare_gan_acgan.png")

def latest_png(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    return files[-1] if files else None

def find_acgan_latest():
    for p in ACGAN_CANDIDATES:
        if os.path.isdir(p):
            f = latest_png(p)
            if f:
                return f
    return None

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def add_label(img: Image.Image, text: str, margin: int = 8):
    """이미지 좌상단에 라벨 텍스트 박스 그리기"""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Pillow >=10 대응 (textsize가 제거됨)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # 하위 버전 Pillow에서는 textsize 여전히 사용 가능
        w, h = draw.textsize(text, font=font)

    pad = 6
    box = [(margin, margin), (margin + w + pad*2, margin + h + pad*2)]
    draw.rectangle(box, fill=(255, 255, 255))
    draw.text((margin + pad, margin + pad), text, fill=(0, 0, 0), font=font)
    return img


def resize_to_same_height(im_left: Image.Image, im_right: Image.Image):
    h = max(im_left.height, im_right.height)
    def resize_h(img, target_h):
        w = int(round(img.width * (target_h / img.height)))
        return img.resize((w, target_h), Image.NEAREST)
    return resize_h(im_left, h), resize_h(im_right, h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-png", type=str, default=None, help="GAN PNG 직접 경로")
    parser.add_argument("--acgan-png", type=str, default=None, help="ACGAN PNG 직접 경로")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="출력 PNG 경로")
    args = parser.parse_args()

    # 1) 입력 PNG 결정
    gan_png = args.gan_png if args.gan_png else latest_png(GAN_DIR)
    if not gan_png:
        raise FileNotFoundError(f"GAN 샘플 PNG을 찾을 수 없습니다: {GAN_DIR}")

    acgan_png = args.acgan_png if args.acgan_png else find_acgan_latest()
    if not acgan_png:
        raise FileNotFoundError(
            "ACGAN 샘플 PNG을 찾을 수 없습니다. 다음 후보 폴더를 확인하세요:\n  - " +
            "\n  - ".join(ACGAN_CANDIDATES)
        )

    # 2) 이미지 로드 및 크기 맞추기
    im_gan = Image.open(gan_png).convert("RGB")
    im_acg = Image.open(acgan_png).convert("RGB")
    im_gan, im_acg = resize_to_same_height(im_gan, im_acg)

    # 3) 라벨 추가 (좌: GAN, 우: ACGAN)
    im_gan = add_label(im_gan, "GAN (Task 1)")
    im_acg = add_label(im_acg, "ACGAN (Task 2)")

    # 4) 캔버스에 합치기
    W = im_gan.width + im_acg.width
    H = max(im_gan.height, im_acg.height)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(im_gan, (0, 0))
    canvas.paste(im_acg, (im_gan.width, 0))

    # 5) 저장
    ensure_dir(args.out)
    canvas.save(args.out)

    print("[OK] Saved:", args.out)
    print(" - Left (GAN):", gan_png)
    print(" - Right (ACGAN):", acgan_png)

if __name__ == "__main__":
    main()
