"""transform_match: the before and after progress photos made as alike as possible WITHOUT touching the physique.
The body is the locked value. Only the lighting (exposure), the skin colour, the clarity and ONE uniform scale change: no warp, no
liquify, no generative AI on the body. Runs on Mac, Windows and Linux: rembg cuts the person out (a mask only, the pixels are never
redrawn) and MediaPipe finds the shoulders, elbows and hips.

usage: transform_match.py <before> <after> <out dir>
writes <out dir>/before.png and after.png (both cut out on one 1440 x 2560 canvas, torso on torso) and match.json (scales, colour moves,
and the callout points on the after's body in 1080 x 1920 reel pixels).

 1. crop the black bars a phone screenshot leaves
 2. cut-out: the person mask, eroded a touch and feathered so no room colour fringes the edge
 3. scale: the before is scaled so its torso (neck to hips) is exactly as long as the after's, so the bones match and the soft tissue
    shows its real change; both are centred on the waist
 4. colour: the before's skin (a torso band) takes the after's mean lightness (exposure) and its skin colour; its lightness SPREAD stays
    its own, so no shadows are drawn on the stomach that were not there
 5. clarity: a light denoise on the before, the same gentle S curve and sharpen on both, a fine fixed grain on a lifted before"""
import _env; _env.need("PIL", "numpy", "rembg", "mediapipe", "pillow_heif")
import json, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageOps

if len(sys.argv) < 4: sys.exit("usage: transform_match.py <before> <after> <out dir>")
SKILL = Path(__file__).resolve().parent.parent
CW, CH = 1440, 2560
TORSO_CANVAS = 840.0          # the after's torso on the canvas: head near the top, knees off the bottom
TARGET = (720.0, 1520.0)      # the torso centre on the canvas
out_dir = Path(sys.argv[3]); out_dir.mkdir(parents=True, exist_ok=True)
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


def fail(s): print(s); sys.exit(1)


def pose_model():
    p = SKILL / ".models" / "pose_landmarker_heavy.task"
    if not p.exists():
        print("FIRST RUN: downloading the body pose model (about 30 MB)...", flush=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_URL, p)
    return p


_cut = _pose = None


def cutter():
    global _cut
    if _cut is None:
        import os
        os.environ.setdefault("U2NET_HOME", str(SKILL / ".models"))
        from rembg import new_session, remove
        s = new_session("u2net_human_seg")   # first run downloads it (about 170 MB)
        _cut = lambda im: remove(im, session=s, only_mask=True)
    return _cut


def poser():
    global _pose
    if _pose is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        _pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model())), running_mode=vision.RunningMode.IMAGE, num_poses=1))
        _pose.mp = mp
    return _pose


class Photo:
    pass


def load(path, label):
    try:
        import pillow_heif; pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        fail(f"CAN'T OPEN: the {label} photo ({path}). Use a JPG, PNG or HEIC.")
    a = np.asarray(im)
    # 1. the black bars of a phone screenshot (rows and columns whose mean is under 6 of 255)
    rows = np.where(a.mean(axis=(1, 2)) > 6)[0]; cols = np.where(a.mean(axis=(0, 2)) > 6)[0]
    if not len(rows) or not len(cols): fail(f"EMPTY PHOTO: the {label} photo is all black.")
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    im = im.crop((int(c0), int(r0), int(c1) + 1, int(r1) + 1))
    p = Photo(); p.im = im; p.w, p.h = im.size; p.crop = [int(c0), int(r0), p.w, p.h]
    # 2. the person mask and the body joints (image pixels, origin top-left). The photo's pixels are never changed here.
    p.mask = np.asarray(cutter()(im).convert("L"))
    if (p.mask > 128).mean() < 0.02: fail(f"NO PERSON FOUND in the {label} photo. Use a photo where the client stands clear of the background.")
    ps = poser()
    res = ps.detect(ps.mp.Image(image_format=ps.mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(im))))
    if not res.pose_landmarks: fail(f"BODY NOT FOUND in the {label} photo. Use a front-on photo with the shoulders and hips in view.")
    L = res.pose_landmarks[0]
    def pt(i):
        q = L[i]
        if (q.visibility or 0) < 0.3 and i in (11, 12, 23, 24): fail(f"BODY NOT FOUND in the {label} photo (shoulders or hips hidden). Use a front-on photo with both in view.")
        return (q.x * p.w, q.y * p.h)
    # joint names kept from the Mac version: "left" is the client's own left
    j = {"left_shoulder_1_joint": pt(11), "right_shoulder_1_joint": pt(12), "left_forearm_joint": pt(13), "right_forearm_joint": pt(14),
         "left_hand_joint": pt(15), "right_hand_joint": pt(16), "left_upLeg_joint": pt(23), "right_upLeg_joint": pt(24), "head_joint": pt(0)}
    sh = ((j["left_shoulder_1_joint"][0] + j["right_shoulder_1_joint"][0]) / 2, (j["left_shoulder_1_joint"][1] + j["right_shoulder_1_joint"][1]) / 2)
    hip = ((j["left_upLeg_joint"][0] + j["right_upLeg_joint"][0]) / 2, (j["left_upLeg_joint"][1] + j["right_upLeg_joint"][1]) / 2)
    j["neck_1_joint"], j["root"] = sh, hip
    p.joints = j
    return p


def torso(p):
    a, b = p.joints["neck_1_joint"], p.joints["root"]; return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def waist(p):
    """the waist's run of mask pixels a fifth of the torso above the hips: its middle is the body's centre, its right end the callout"""
    root = p.joints["root"]; y = int(max(0, min(p.h - 1, root[1] - 0.2 * torso(p)))); row = p.mask[y] > 128
    x = int(max(0, min(p.w - 1, root[0])))
    if not row[x]:
        idx = np.where(row)[0]
        if not len(idx): return (root[0] - 50, root[0] + 50, float(y))
        x = int(idx[np.argmin(np.abs(idx - x))])
    l = x
    while l > 0 and row[l - 1]: l -= 1
    r = x
    while r < p.w - 1 and row[r + 1]: r += 1
    return (float(l), float(r), float(y))


# ---- sRGB <-> Lab (D65), so the colour move is on lightness and skin colour separately
def to_lab(rgb):
    c = rgb / 255.0; c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    X = (0.4124 * c[..., 0] + 0.3576 * c[..., 1] + 0.1805 * c[..., 2]) / 0.95047
    Y = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
    Z = (0.0193 * c[..., 0] + 0.1192 * c[..., 1] + 0.9505 * c[..., 2]) / 1.08883
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    fx, fy, fz = f(X), f(Y), f(Z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def from_lab(lab):
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    inv = lambda t: np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    X, Y, Z = inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883
    rgb = np.stack([3.2406 * X - 1.5372 * Y - 0.4986 * Z, -0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0.0557 * X - 0.2040 * Y + 1.0570 * Z], -1)
    rgb = np.clip(rgb, 0, 1)
    return 255 * np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def skin_stats(lab, p):
    """the skin band: inside the mask, under the neck to over the hips, the middle 60 % of the shoulder span (away from the arms)"""
    t, n, r = torso(p), p.joints["neck_1_joint"], p.joints["root"]
    xs = sorted([p.joints["left_shoulder_1_joint"][0], p.joints["right_shoulder_1_joint"][0]]); span = xs[1] - xs[0]
    y0, y1 = max(0, int(n[1] + 0.06 * t)), min(p.h, int(r[1] - 0.09 * t))
    x0, x1 = max(0, int(xs[0] + 0.2 * span)), min(p.w, int(xs[1] - 0.2 * span))
    sel = lab[y0:y1, x0:x1][p.mask[y0:y1, x0:x1] > 200] if y1 > y0 and x1 > x0 else np.zeros((0, 3))
    if len(sel) < 500: fail("SKIN NOT FOUND: the torso isn't clear in one of the photos. Use photos where the stomach and chest show.")
    return sel.mean(0), np.maximum(sel.std(0), 1e-3)


before, after = load(sys.argv[1], "before"), load(sys.argv[2], "after")

# 5a. a light denoise on the before (usually the dimmer, noisier phone photo) before anything is measured
b_img = Image.blend(before.im, before.im.filter(ImageFilter.MedianFilter(3)), 0.5)
b_lab, a_lab = to_lab(np.asarray(b_img).astype(np.float64)), to_lab(np.asarray(after.im).astype(np.float64))
mb, sb = skin_stats(b_lab, before); ma, sa = skin_stats(a_lab, after)
# 4. lightness: a shift only (exposure). Skin colour (a, b): the after's mean and spread.
ratio = np.array([1.0, sa[1] / sb[1], sa[2] / sb[2]])
b_rgb = from_lab((b_lab - mb) * ratio + ma)
a_rgb = np.asarray(after.im).astype(np.float64)

x = np.arange(256) / 255
GENTLE_S = np.clip(255 * (x + 0.06 * (x - 0.5) * (1 - np.abs(2 * x - 1))), 0, 255)


def finish(rgb):   # one shared finish: the gentle S curve and a touch of warmth
    o = GENTLE_S[np.clip(rgb.round(), 0, 255).astype(np.uint8)]
    o[..., 0] *= 1.02; o[..., 2] *= 0.985
    return np.clip(o.round(), 0, 255).astype(np.uint8)


b_out = Image.fromarray(finish(b_rgb)).filter(ImageFilter.UnsharpMask(radius=2.2, percent=45, threshold=0))
a_out = Image.fromarray(finish(a_rgb)).filter(ImageFilter.UnsharpMask(radius=2.2, percent=35, threshold=0))
lift = float(ma[0] - mb[0])
if abs(lift) > 8:   # a big exposure move stretches the photo's 8-bit steps into bands: a fine fixed grain breaks them
    g = np.random.default_rng(7).normal(0, 2.6, (before.h, before.w, 1))
    b_out = Image.fromarray(np.clip(np.asarray(b_out).astype(np.float64) + g, 0, 255).round().astype(np.uint8))

# 3. the uniform scale and the placement, then the cut-out onto the canvas
sA = TORSO_CANVAS / torso(after); sB = sA * torso(after) / torso(before)


def centre(p):
    w = waist(p); return ((w[0] + w[1]) / 2, (p.joints["neck_1_joint"][1] + p.joints["root"][1]) / 2)


def place(img, p, s, erode, name):
    c = centre(p); tx, ty = TARGET[0] - s * c[0], TARGET[1] - s * c[1]
    m = Image.fromarray(p.mask).filter(ImageFilter.MinFilter(2 * erode + 1)).filter(ImageFilter.GaussianBlur(1.4))
    cut = img.convert("RGBA"); cut.putalpha(m)
    size = (max(1, round(p.w * s)), max(1, round(p.h * s)))
    cut = cut.resize(size, Image.LANCZOS)
    canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0)); canvas.paste(cut, (round(tx), round(ty)), cut)
    canvas.save(out_dir / f"{name}.png")
    return tx, ty


place(b_out, before, sB, 3, "before")
tx, ty = place(a_out, after, sA, 2, "after")


def reel(q): return [round((sA * q[0] + tx) * 0.75, 1), round((sA * q[1] + ty) * 0.75, 1)]


J = after.joints
shs = sorted([J["left_shoulder_1_joint"], J["right_shoulder_1_joint"]], key=lambda v: v[0])   # image left, image right
neck, root, w, T = J["neck_1_joint"], J["root"], waist(after), torso(after)
pts = {k: reel(v) for k, v in J.items()}
pts["shoulder"] = reel((shs[0][0] + 0.03 * T, shs[0][1] + 0.015 * T))   # just inside the image-left shoulder
pts["chest"] = reel((shs[1][0] + (neck[0] - shs[1][0]) * 0.45 + (root[0] - neck[0]) * 0.22, shs[1][1] + (neck[1] - shs[1][1]) * 0.45 + (root[1] - neck[1]) * 0.22))
pts["abs"] = reel((neck[0] + (root[0] - neck[0]) * 0.62, neck[1] + (root[1] - neck[1]) * 0.62))
pts["waist"] = reel((w[1] - 0.035 * T, w[2]))                          # just inside the waist's edge
pts["head_top"] = reel((neck[0], J["head_joint"][1] - 0.25 * T))
info = {"canvas": [CW, CH], "reel": [1080, 1920],
        "before": {"crop": before.crop, "torso_px": torso(before), "scale": sB}, "after": {"crop": after.crop, "torso_px": torso(after), "scale": sA},
        "skin_lab": {"before_mean": mb.tolist(), "before_spread": sb.tolist(), "after_mean": ma.tolist(), "after_spread": sa.tolist(), "exposure_shift_L": lift},
        "anchors": pts}
(out_dir / "match.json").write_text(json.dumps(info, indent=1, sort_keys=True))
print(f"ok before scale {sB:.3f} after scale {sA:.3f} exposure shift L {lift:.1f}")
