!pip install --quiet opencv-python-headless matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# ---------- 유틸리티 ----------
def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def normalize_img(img):
    a = img.astype(np.float32)
    a -= a.min()
    if a.max() != 0:
        a /= a.max()
    return a

def show_image(ax, img, title=""):
    ax.axis('off')
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=11)

def make_pixel_grid(img_float_rgb, grid_n=14):
    # img_float_rgb: HxWx3 in range [0,1]
    h, w, _ = img_float_rgb.shape
    gh = max(1, h // grid_n)
    gw = max(1, w // grid_n)
    grid = np.zeros((grid_n, grid_n), dtype=np.float32)
    mean_ch = img_float_rgb.mean(axis=2)
    for i in range(grid_n):
        for j in range(grid_n):
            y1 = i * gh
            x1 = j * gw
            y2 = (i+1) * gh if i < grid_n-1 else h
            x2 = (j+1) * gw if j < grid_n-1 else w
            block = mean_ch[y1:y2, x1:x2]
            grid[i, j] = block.mean() if block.size>0 else 0.0
    return grid

def conv2d_valid_singlechannel(img_ch, kernel):
    kh, kw = kernel.shape
    ih, iw = img_ch.shape
    oh = ih - kh + 1
    ow = iw - kw + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((ih, iw), dtype=np.float32)
    out = np.zeros((oh, ow), dtype=np.float32)
    for y in range(oh):
        for x in range(ow):
            patch = img_ch[y:y+kh, x:x+kw]
            out[y,x] = np.sum(patch * kernel)
    return out

# ---------- 업로드 및 기본 처리 ----------
print("이미지 파일 업로드 (바탕화면에서 선택).")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No files uploaded")
fname = next(iter(uploaded.keys()))
bgr = cv2.imdecode(np.frombuffer(uploaded[fname], np.uint8), cv2.IMREAD_COLOR)
orig_rgb = to_rgb(bgr)

# 리사이즈 (모델 입력 크기)
resize_h, resize_w = 224, 224
resized_rgb = cv2.resize(orig_rgb, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

# Pixel-grid (visualization of block-wise mean)
resized_vis = normalize_img(resized_rgb.astype(np.float32) / 255.0)
grid_n = 14
grid = make_pixel_grid(resized_vis, grid_n=grid_n)
grid_disp = normalize_img(grid)

# ---------- 패딩 시각화 (중앙 배치) ----------
canvas_h, canvas_w = 260, 260
pad_vert = canvas_h - resize_h
pad_h_top = max(0, pad_vert // 2)
pad_h_bottom = max(0, pad_vert - pad_h_top)
pad_horiz = canvas_w - resize_w
pad_w_left = max(0, pad_horiz // 2)
pad_w_right = max(0, pad_horiz - pad_w_left)

def pad_and_overlay(orig_rgb, top, bottom, left, right, pad_value=0, borderType=cv2.BORDER_CONSTANT):
    h, w, _ = orig_rgb.shape
    if borderType == cv2.BORDER_CONSTANT:
        padded = cv2.copyMakeBorder(orig_rgb, top, bottom, left, right, borderType, value=[pad_value]*3)
    else:
        padded = cv2.copyMakeBorder(orig_rgb, top, bottom, left, right, borderType)
    out = padded.copy()
    cv2.rectangle(out, (left, top), (left+w-1, top+h-1), (255,255,255), 2)
    return padded, out

p_const, p_const_overlay = pad_and_overlay(resized_rgb, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, pad_value=0, borderType=cv2.BORDER_CONSTANT)
p_reflect, p_reflect_overlay = pad_and_overlay(resized_rgb, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, borderType=cv2.BORDER_REFLECT)
p_replicate, p_repl_overlay = pad_and_overlay(resized_rgb, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, borderType=cv2.BORDER_REPLICATE)

# ---------- 컨볼루션 연산 (채널별, 다양한 border) ----------
kernels = {
    "Sobel X (edge)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
    "Gaussian Blur": (1/16.0)*np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32)
}
border_modes = {
    "valid (no pad)": None,
    "same - replicate": cv2.BORDER_REPLICATE,
    "same - reflect": cv2.BORDER_REFLECT,
    "same - constant 0": cv2.BORDER_CONSTANT
}
conv_results = {}
for kname, kernel in kernels.items():
    conv_results[kname] = {}
    for bname, bmode in border_modes.items():
        if bname == "valid (no pad)":
            channels = []
            for c in range(3):
                ch = resized_rgb[:,:,c].astype(np.float32)
                out_valid = conv2d_valid_singlechannel(ch, kernel)
                oh, ow = out_valid.shape
                pad_top = (resized_rgb.shape[0] - oh)//2
                pad_left = (resized_rgb.shape[1] - ow)//2
                out_pad = np.zeros_like(ch)
                out_pad[pad_top:pad_top+oh, pad_left:pad_left+ow] = out_valid
                channels.append(out_pad)
            conv_img = np.stack(channels, axis=2)
            conv_results[kname][bname] = conv_img
        else:
            channels = []
            for c in range(3):
                ch = resized_rgb[:,:,c].astype(np.float32)
                filtered = cv2.filter2D(ch, ddepth=-1, kernel=kernel, borderType=bmode)
                channels.append(filtered)
            conv_img = np.stack(channels, axis=2)
            conv_results[kname][bname] = conv_img

# ---------- 중앙 3x3 패치의 컨볼루션 스텝 시각화 ----------
center_y, center_x = resize_h//2, resize_w//2
ps = 3
py1 = center_y - ps//2
px1 = center_x - ps//2
patch = resized_rgb[py1:py1+ps, px1:px1+ps].astype(np.float32)
kernel = kernels["Sobel X (edge)"]
patch_mean = patch.mean(axis=2)
elementwise = patch_mean * kernel
elementwise_sum = elementwise.sum()

# ---------- 모델 예측 (예측 확률 패널) ----------
inp_arr = keras_image.img_to_array(resized_rgb)
x = np.expand_dims(inp_arr, 0)
x_pre = preprocess_input(x.copy())
model = mobilenet_v2.MobileNetV2(weights='imagenet')
preds = model.predict(x_pre)
topk = 8
decoded = decode_predictions(preds, top=topk)[0]
labels = [d[1] for d in decoded]
probs = [float(d[2]) for d in decoded]
pred_label = labels[0]
pred_prob = probs[0]

# ---------- 세로(column) 레이아웃으로 표시 (Preprocessed(mean) 제거, pixel-grid+pred probs 포함) ----------
panels = [
    ("Original", normalize_img(orig_rgb.astype(np.float32)/255.0)),
    ("Resized 224x224", normalize_img(resized_rgb.astype(np.float32)/255.0)),
    ("Pixel grid (block means)", grid_disp),
    ("Padded - constant(0) overlay", normalize_img(p_const_overlay.astype(np.float32)/255.0)),
    ("Padded - reflect overlay", normalize_img(p_reflect_overlay.astype(np.float32)/255.0)),
    ("Padded - replicate overlay", normalize_img(p_repl_overlay.astype(np.float32)/255.0)),
]
# Add one conv result per kernel (use array directly; do NOT call np.clip with None)
for kname in kernels.keys():
    conv_img_same_reflect = conv_results[kname]["same - reflect"]
    panels.append((f"Conv ({kname}) - same(reflect)", normalize_img(conv_img_same_reflect)))

panels.append(("ReLU example (Sharpen, reflect)", normalize_img(np.maximum(conv_results["Sharpen"]["same - reflect"], 0))))

nrows = len(panels) + 2  # one for kernel numbers, one for patch-step
fig = plt.figure(figsize=(6, 3 * nrows))

row = 1
for title, img in panels:
    ax = fig.add_subplot(nrows, 1, row)
    # Pixel grid needs numeric overlay
    if title == "Pixel grid (block means)":
        im = ax.imshow(img, cmap='magma', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        # overlay numeric values
        gn = img.shape[0]
        for r in range(gn):
            for c in range(gn):
                val = img[r, c]
                txt = f"{val:.2f}"
                color = "white" if val < 0.5 else "black"
                ax.text(c, r, txt, ha='center', va='center', color=color, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    else:
        show_image(ax, img, title=title)
    row += 1

# Kernel numeric display (Sobel X)
axk = fig.add_subplot(nrows, 1, row)
axk.axis('off')
axk.set_title("Kernel (Sobel X) numeric", fontsize=11)
kernel_vis = np.round(kernel.astype(np.float32), 3)
axk.imshow(kernel_vis, cmap='coolwarm', vmin=-np.max(np.abs(kernel_vis)), vmax=np.max(np.abs(kernel_vis)))
# numeric overlay
h, w = kernel_vis.shape
for i in range(h):
    for j in range(w):
        axk.text(j, i, f"{kernel_vis[i,j]:.0f}", ha='center', va='center', color='black', fontsize=10)
row += 1

# Patch * kernel elementwise and sum display
axe = fig.add_subplot(nrows, 1, row)
axe.axis('off')
axe.set_title("Patch (mean) * Kernel  → elementwise and sum", fontsize=11)
pm = np.round(patch_mean, 1)
el = np.round(elementwise, 2)
left = (pm - pm.min())
if left.max() != 0:
    left /= left.max()
left = (left*255).astype(np.uint8)
mid = (kernel - kernel.min())
if mid.max() != 0:
    mid = (mid / mid.max()*255).astype(np.uint8)
else:
    mid = (kernel+127).astype(np.uint8)
right = el.copy()
right = right - right.min()
if right.max() != 0:
    right = (right / right.max()*255).astype(np.uint8)
else:
    right = (right*0).astype(np.uint8)
left_rgb = cv2.cvtColor(cv2.resize(left, (90,90)), cv2.COLOR_GRAY2RGB)
mid_rgb = cv2.cvtColor(cv2.resize(mid, (90,90)), cv2.COLOR_GRAY2RGB)
right_rgb = cv2.cvtColor(cv2.resize(right, (90,90)), cv2.COLOR_GRAY2RGB)
pad = 4
comp = np.hstack([left_rgb, np.zeros((90,pad,3),dtype=np.uint8), mid_rgb, np.zeros((90,pad,3),dtype=np.uint8), right_rgb])
axe.imshow(comp)
# overlay numbers
for i in range(3):
    for j in range(3):
        y = int((i+0.5)*(90/3))
        x = int((j+0.5)*(90/3))
        axe.text(x, y, f"{pm[i,j]:.1f}", ha='center', va='center', color='white', fontsize=8)
off_x = 90 + pad
for i in range(3):
    for j in range(3):
        y = int((i+0.5)*(90/3))
        x = off_x + int((j+0.5)*(90/3))
        axe.text(x, y, f"{kernel[i,j]:.0f}", ha='center', va='center', color='black', fontsize=8)
off_x2 = off_x + 90 + pad
for i in range(3):
    for j in range(3):
        y = int((i+0.5)*(90/3))
        x = off_x2 + int((j+0.5)*(90/3))
        axe.text(x, y, f"{el[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
axe.text(comp.shape[1]*0.5, 92, f"Sum = {elementwise_sum:.2f}", ha='center', va='top', fontsize=11, color='black')

plt.tight_layout()
plt.show()

# Prediction probs panel shown after visualization (console + small figure)
fig2 = plt.figure(figsize=(6,3))
axp = fig2.add_subplot(1,1,1)
y_pos = np.arange(len(labels))
axp.barh(y_pos, probs[::-1], color='tab:blue')
axp.set_yticks(y_pos)
axp.set_yticklabels(labels[::-1], fontsize=10)
axp.set_xlim(0, 1)
axp.invert_yaxis()
axp.set_xlabel("Probability")
axp.set_title("Top-8 predictions")
plt.suptitle(f"{fname}  →  Predicted: {pred_label} ({pred_prob:.2f})", fontsize=12)
plt.tight_layout()
plt.show()

print("Top-3 predictions:")
for i, d in enumerate(decoded[:3], start=1):
    print(f"{i}. {d[1]}: {d[2]:.4f}")
print(f"Recognized: {pred_label} ({pred_prob:.2f})")
