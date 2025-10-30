# fix_facades_structure.py
# facades 구조가 facades/facades/... 처럼 중첩일 때 평탄화
import os
import shutil

def count_images(p):
    if not os.path.isdir(p): return 0
    return sum(f.lower().endswith((".jpg",".jpeg",".png",".bmp")) for f in os.listdir(p))

def fix(dataroot="./facades"):
    print("[fix] dataroot =", dataroot)
    nested = os.path.join(dataroot, "facades", "facades")
    target = os.path.join(dataroot, "facades")
    if os.path.isdir(nested):
        print(f"[fix] flatten: {nested} -> {target}")
        for sub in ["train","test","val"]:
            src = os.path.join(nested, sub)
            dst = os.path.join(target, sub)
            if os.path.isdir(dst): shutil.rmtree(dst)
            shutil.move(src, dst)
        shutil.rmtree(nested)

    tr = os.path.join(target, "train")
    te = os.path.join(target, "test")
    print(f"[fix] #train: {count_images(tr)}  #test: {count_images(te)}")
    if count_images(tr)==0 or count_images(te)==0:
        print("[fix] 경로가 다르면 직접 지정: python fix_facades_structure.py C:\\path\\to\\facades")

if __name__ == "__main__":
    import sys
    dataroot = sys.argv[1] if len(sys.argv)>1 else "./facades"
    fix(dataroot)
