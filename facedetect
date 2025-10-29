import sys, os, cv2
import matplotlib.pyplot as plt

# allowed image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image_file(path):
    _, ext = os.path.splitext(path.lower())
    return ext in IMG_EXTS and os.path.isfile(path)

def try_colab_upload():
    try:
        from google.colab import files
        uploaded = files.upload()
        return list(uploaded.keys())
    except Exception:
        return []

# gather candidate filenames from argv, filtering out flags and non-image paths
args = sys.argv[1:]
candidates = [a for a in args if not a.startswith('-') and is_image_file(a)]

# if no valid file paths from argv, try Colab upload or prompt
if not candidates:
    uploaded = try_colab_upload()
    if uploaded:
        candidates = uploaded
    else:
        # fallback: ask user to type a path (useful in some notebook environments)
        inp = input("이미지 파일 경로를 입력하세요 (여러개는 쉼표로 구분), 또는 Enter 취소: ").strip()
        if inp:
            for p in [s.strip() for s in inp.split(',')]:
                if is_image_file(p):
                    candidates.append(p)
                else:
                    print(f"이미지로 사용할 수 없음 또는 파일 없음: {p}")

if not candidates:
    raise RuntimeError("처리할 이미지 파일이 지정되지 않았습니다. 커맨드라인 인자 또는 업로드로 이미지 파일을 제공하세요.")

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for fname in candidates:
    # support both Colab-uploaded files and local paths
    img = cv2.imread(fname)
    if img is None:
        print(f"파일을 열 수 없음(이미지 아님/손상): {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{os.path.basename(fname)} — Faces: {len(faces)}")
    plt.axis('off')
    plt.show()
    print(f"{fname}: {len(faces)} faces detected. Boxes: {faces.tolist() if len(faces) else []}")
