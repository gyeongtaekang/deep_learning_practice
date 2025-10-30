import os, glob, argparse, torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from importlib.util import spec_from_file_location, module_from_spec

# ---------- 경로 설정 ----------
HERE = os.path.dirname(os.path.abspath(__file__))
PIX_DIR = os.path.normpath(os.path.join(HERE, "..", "pix2pix"))      # implementations/pix2pix
CYC_DIR = HERE                                                        # implementations/cyclegan

def _load_module(module_name: str, file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"[ERR] module file not found: {file_path}")
    spec = spec_from_file_location(module_name, file_path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---------- 각 모델의 models.py를 '파일경로로' 직접 로드 ----------
pix_models = _load_module("pix2pix_models", os.path.join(PIX_DIR, "models.py"))
cyc_models = _load_module("cyclegan_models", os.path.join(CYC_DIR, "models.py"))

Pix2PixGenerator = pix_models.GeneratorUNet           # U-Net (pix2pix)
GeneratorResNet = cyc_models.GeneratorResNet          # ResNet (cyclegan)

# ---------- 유틸 ----------
def latest_ckpt(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_pix2pix_generator(pix_ckpt, device, in_ch=3, out_ch=3):
    netG = Pix2PixGenerator(in_channels=in_ch, out_channels=out_ch).to(device)
    if pix_ckpt is None:
        cand = []
        cand += glob.glob(os.path.join(PIX_DIR, "saved_models", "*.pth"))
        cand += glob.glob(os.path.join(PIX_DIR, "saved_models", "**", "*.pth"), recursive=True)
        pix_ckpt = cand[-1] if cand else None
    if pix_ckpt is None:
        raise FileNotFoundError("Pix2Pix 체크포인트(.pth)를 찾지 못했습니다. --pix2pix_ckpt 로 지정하세요.")
    state = torch.load(pix_ckpt, map_location=device)
    # 다양한 포맷 대응
    try:
        netG.load_state_dict(state)
    except Exception:
        ok = False
        if isinstance(state, dict):
            for k in ["state_dict", "generator", "netG", "model"]:
                if k in state:
                    netG.load_state_dict(state[k]); ok = True; break
        if not ok: raise
    print(f"[pix2pix] loaded: {pix_ckpt}")
    netG.eval()
    return netG

def load_cyclegan_generator(cyc_ckpt, device, n_res=9, in_ch=3):
    netG = GeneratorResNet((in_ch, 256, 256), n_res).to(device)
    if cyc_ckpt is None:
        cyc_dir = os.path.join(CYC_DIR, "saved_models", "facades_cyclegan")
        cyc_ckpt = latest_ckpt(os.path.join(cyc_dir, "G_AB_*.pth"))
        if cyc_ckpt is None:
            cands = glob.glob(os.path.join(CYC_DIR, "saved_models", "**", "G_AB_*.pth"), recursive=True)
            cyc_ckpt = cands[-1] if cands else None
    if cyc_ckpt is None:
        raise FileNotFoundError("CycleGAN G_AB 체크포인트(.pth)를 찾지 못했습니다. --cyclegan_ckpt 로 지정하세요.")
    state = torch.load(cyc_ckpt, map_location=device)
    netG.load_state_dict(state if isinstance(state, dict) else state.state_dict())
    print(f"[cyclegan] loaded: {cyc_ckpt}")
    netG.eval()
    return netG

def load_facades_test_pairs(test_dir):
    files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
    if not files:
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {test_dir}")
    return files

def split_A_B(img_pil):
    w, h = img_pil.size
    A = img_pil.crop((0, 0, w // 2, h)).convert("RGB")
    B = img_pil.crop((w // 2, 0, w, h)).convert("RGB")
    return A, B

def to_tensor_01(pil, size=256):
    return T.Compose([T.Resize((size, size)), T.ToTensor()])(pil)

def to_tensor_norm(pil, size=256):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])(pil)

# ---------- 메인 ----------
@torch.no_grad()
def run_compare(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 모델 로드
    netG_pix = load_pix2pix_generator(args.pix2pix_ckpt, device)
    netG_cyc = load_cyclegan_generator(args.cyclegan_ckpt, device, n_res=args.n_residual_blocks)

    # 2) 테스트 이미지 (pix2pix 형식: A|B 한 장에)
    test_dir = os.path.normpath(args.pix2pix_facades_test)
    files = load_facades_test_pairs(test_dir)
    print(f"[data] #test files: {len(files)}")

    samples = files[:args.num]
    rows = []
    for fp in samples:
        img = Image.open(fp).convert("RGB")
        A, B = split_A_B(img)

        A_in = to_tensor_norm(A, args.size).unsqueeze(0).to(device)

        # A -> B
        fake_B_pix = netG_pix(A_in)
        fake_B_cyc = netG_cyc(A_in)

        visA  = to_tensor_01(A, args.size)
        visB  = to_tensor_01(B, args.size)
        visP  = (fake_B_pix.clamp(-1, 1) * 0.5 + 0.5).cpu().squeeze(0)
        visC  = (fake_B_cyc.clamp(-1, 1) * 0.5 + 0.5).cpu().squeeze(0)

        rows.append(torch.stack([visA, visP, visC, visB], dim=0))  # A | Pix2Pix | CycleGAN | GT

    grid = make_grid(torch.cat(rows, dim=0), nrow=4, padding=4, normalize=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_image(grid, args.out)
    print(f"[ok] saved compare panel -> {args.out}")
    print("열 순서: Input A | Pix2Pix | CycleGAN | GroundTruth B")

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pix2pix_ckpt", type=str, default=None, help="pix2pix .pth (optional)")
    ap.add_argument("--cyclegan_ckpt", type=str, default=None, help="cyclegan G_AB_*.pth (optional)")
    ap.add_argument("--pix2pix_facades_test", type=str,
                    default=os.path.join(PIX_DIR, "facades", "facades", "test"),
                    help="pix2pix용 facades test 폴더 (A|B in one image)")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "pix2pix_vs_cyclegan.png"))
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n_residual_blocks", type=int, default=9)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_compare(args)
