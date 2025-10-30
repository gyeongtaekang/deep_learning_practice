# plot_acgan_losses_v4.py
# - PowerShell Tee-Object가 저장한 UTF-16 LE 로그도 자동 감지하여 파싱
# - 정규식에 의존하지 않고 Epoch/Batch/D loss/G loss 토큰을 스캔
# - 결과: acgan_loss_curve_iter.png, acgan_loss_curve_epoch.png, acgan_losses.csv

import os, sys, csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

HERE = os.path.dirname(__file__)
LOG_PATH = os.path.join(HERE, "train_log.txt")

def read_text_auto(path):
    """
    UTF-8 시도 -> UTF-16 LE 시도 -> UTF-16 BE 시도 -> cp949/latin-1 시도
    """
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp949", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
            continue
    # 최후의 보루: 바이너리 읽어서 바이트를 유니코드로 그대로 매핑
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-16-le", errors="ignore")
    except Exception:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            print(f"[WARN] 인코딩 감지 실패: {last_err}")
            return data.decode("latin-1", errors="ignore")

def to_float(tok: str):
    s = "".join(ch for ch in tok if (ch.isdigit() or ch in ".-+eE"))
    if not s: 
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_records(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    recs = []  # (epoch, batch, D, G)
    for ln in lines:
        if ("Epoch" not in ln) or ("Batch" not in ln) or ("D loss" not in ln) or ("G loss" not in ln):
            continue
        try:
            i_ep = ln.index("Epoch")
            i_ba = ln.index("Batch", i_ep)
            i_d  = ln.index("D loss", i_ba)
            i_g  = ln.index("G loss", i_d)
        except ValueError:
            continue

        # Epoch 구간
        seg_ep = ln[i_ep:i_ba]
        ep = None
        for tok in seg_ep.replace("["," ").replace("]"," ").replace("/"," ").split():
            if tok.isdigit():
                ep = int(tok); break
        if ep is None:
            continue

        # Batch 구간
        seg_ba = ln[i_ba:i_d]
        ba = None
        for tok in seg_ba.replace("["," ").replace("]"," ").replace("/"," ").split():
            if tok.isdigit():
                ba = int(tok); break
        if ba is None:
            continue

        # D loss 값
        seg_d = ln[i_d:i_g]
        d = None
        if ":" in seg_d:
            parts = seg_d.split(":", 1)[1].replace(",", " ").split()
            for t in parts:
                v = to_float(t)
                if v is not None:
                    d = v; break
        if d is None:
            continue

        # G loss 값
        seg_g = ln[i_g:]
        g = None
        if ":" in seg_g:
            parts = seg_g.split(":", 1)[1].replace(",", " ").split()
            for t in parts:
                v = to_float(t)
                if v is not None:
                    g = v; break
        if g is None:
            continue

        recs.append((ep, ba, d, g))
    return recs

def main():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        print(f"[ERR] 로그 파일이 없거나 비어있습니다: {LOG_PATH}")
        sys.exit(1)

    text = read_text_auto(LOG_PATH)
    records = parse_records(text)

    if not records:
        print("[ERR] 로그에서 손실을 찾지 못했습니다.")
        # 디버그: 앞 10줄 출력
        sample = "\n".join(text.splitlines()[:10])
        print("[DEBUG] 파일 앞 10줄:\n", sample)
        sys.exit(1)

    # CSV 저장
    csv_path = os.path.join(HERE, "acgan_losses.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch","batch","D_loss","G_loss"])
        for ep, ba, d, g in records:
            w.writerow([ep, ba, d, g])

    # Iteration 곡선
    g_iter = [r[3] for r in records]
    d_iter = [r[2] for r in records]
    plt.figure(figsize=(8,5))
    plt.plot(g_iter, label="G loss (iter)")
    plt.plot(d_iter, label="D loss (iter)", alpha=0.7)
    plt.title("ACGAN Training Loss (per iteration)")
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    out_iter = os.path.join(HERE, "acgan_loss_curve_iter.png")
    plt.savefig(out_iter, dpi=150); plt.close()

    # Epoch 평균 곡선
    sumD, sumG, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    for ep, ba, d, g in records:
        sumD[ep] += d; sumG[ep] += g; cnt[ep] += 1
    epochs = sorted(cnt.keys())
    d_ep = [sumD[e]/cnt[e] for e in epochs]
    g_ep = [sumG[e]/cnt[e] for e in epochs]
    plt.figure(figsize=(8,5))
    plt.plot(epochs, g_ep, label="G loss (epoch)")
    plt.plot(epochs, d_ep, label="D loss (epoch)", alpha=0.7)
    plt.title("ACGAN Training Loss (per epoch)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    out_ep = os.path.join(HERE, "acgan_loss_curve_epoch.png")
    plt.savefig(out_ep, dpi=150); plt.close()

    print(f"Saved: {out_iter}")
    print(f"Saved: {out_ep}")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
