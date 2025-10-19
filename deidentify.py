#!/usr/bin/env python3
# deidentify.py
# Photo/Video de-identifier: blur/pixelate/box faces and text (incl. plates)

import os
import json
import cv2
import argparse
import time
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps

# ---- Face detector (insightface / retinaface) ----
from insightface.app import FaceAnalysis

# ---- Text detector (PaddleOCR detector only) ----
from paddleocr import PaddleOCR


# Utilities

def read_image_with_exif(path):
    """Read an image and respect EXIF orientation."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def save_preview_side_by_side(orig_bgr, red_bgr, out_path_png):
    """Save a before/after preview PNG for quick eyeballing."""
    h = max(orig_bgr.shape[0], red_bgr.shape[0])
    w = orig_bgr.shape[1] + red_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:orig_bgr.shape[0], :orig_bgr.shape[1]] = orig_bgr
    canvas[:red_bgr.shape[0], orig_bgr.shape[1]:] = red_bgr
    cv2.imwrite(out_path_png, canvas)

def to_int_box(xyxy, W, H):
    x1, y1, x2, y2 = map(int, xyxy)
    return [max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)]

def polygon_to_box(poly, W, H):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return to_int_box([min(xs), min(ys), max(xs), max(ys)], W, H)

def pad_box(box, W, H, pad_px=0):
    if pad_px <= 0:
        return box
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - pad_px),
        max(0, y1 - pad_px),
        min(W - 1, x2 + pad_px),
        min(H - 1, y2 + pad_px),
    ]

def apply_redaction(img, box, mode="blur", strength=25):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    if mode == "blur":
        k = max(3, strength | 1)
        red = cv2.GaussianBlur(roi, (k, k), 0)
    elif mode == "pixelate":
        h, w = roi.shape[:2]
        f = max(4, strength // 2)
        red = cv2.resize(
            cv2.resize(roi, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_LINEAR),
            (w, h), interpolation=cv2.INTER_NEAREST
        )
    else:
        red = np.zeros_like(roi)
    img[y1:y2, x1:x2] = red

def merge_boxes(boxes, iou_thr=0.3):
    """NMS to avoid double-masking same region."""
    if not boxes:
        return []
    b = np.array(boxes, dtype=float)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return [boxes[i] for i in keep]


# Detectors

def build_face_detector(ctx_id=-1, det_size=(640, 640), det_thresh=0.6):
    """
    ctx_id: -1 for CPU, 0+ for GPU.
    det_size: larger => higher recall, slower. Try 768 or 896 for tough scenes.
    det_thresh: face confidence threshold.
    """
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    return app

def build_text_detector(use_gpu=False, det_db_thresh=0.3):
    """
    PaddleOCR detector-only. 
    det_db_thresh ~0.3 is a good default.
    """
    ocr = PaddleOCR(
        use_angle_cls=False, lang='en',
        det=True, rec=False, use_gpu=use_gpu, show_log=False,
        det_db_thresh=det_db_thresh
    )
    return ocr


# Per-frame redaction

def redact_frame(frame, face_app, text_ocr, mode, strength,
                 include_text=True, pad_px=6, face_thr=0.6, text_thr=0.3):
    H, W = frame.shape[:2]
    boxes = []

    faces = face_app.get(frame)
    for f in faces:
        score = getattr(f, "det_score", 1.0)
        if score < face_thr:
            continue
        boxes.append(pad_box(to_int_box(f.bbox, W, H), W, H, pad_px))

    if include_text and text_ocr is not None:
        result = text_ocr.ocr(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cls=False, det=True, rec=False)
        if result and len(result[0]) > 0:
            for det in result[0]:
                poly = det[0]
                score = 1.0
                if len(det) > 1 and det[1] is not None:
                    if isinstance(det[1], (list, tuple)) and len(det[1]) > 1:
                        score = det[1][1] if isinstance(det[1][1], (int, float)) else score
                    elif isinstance(det[1], (int, float)):
                        score = det[1]
                if poly is None or (score is not None and score < text_thr):
                    continue
                box = polygon_to_box(poly, W, H)
                boxes.append(pad_box(box, W, H, pad_px))

    boxes = merge_boxes(boxes, iou_thr=0.2)
    for b in boxes:
        apply_redaction(frame, b, mode=mode, strength=strength)
    return frame, boxes


# Media processing

def process_image(path_in, path_out, face_app, text_ocr, mode, strength,
                  include_text, pad_px, face_thr, text_thr, preview=False):
    orig = read_image_with_exif(path_in)
    img = orig.copy()
    red, boxes = redact_frame(
        img, face_app, text_ocr, mode, strength,
        include_text=include_text, pad_px=pad_px,
        face_thr=face_thr, text_thr=text_thr
    )
    cv2.imwrite(path_out, red)
    if preview:
        prev = os.path.splitext(path_out)[0] + "_preview.png"
        save_preview_side_by_side(orig, red, prev)
    return boxes, (orig.shape[1], orig.shape[0])  # (W, H)

def process_video(path_in, path_out, face_app, text_ocr, mode, strength,
                  include_text, pad_px, face_thr, text_thr, json_path=None):
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        raise FileNotFoundError(path_in)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N_FR = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path_out, fourcc, FPS, (W, H))
    meta = {"source": os.path.basename(path_in), "width": W, "height": H, "fps": FPS, "frames": []}

    pbar = tqdm(total=N_FR, desc="Redacting", unit="frame")
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, boxes = redact_frame(
            frame, face_app, text_ocr, mode, strength,
            include_text=include_text, pad_px=pad_px,
            face_thr=face_thr, text_thr=text_thr
        )
        out.write(frame)
        if json_path:
            meta["frames"].append({"index": fidx, "boxes": [list(map(int, b)) for b in boxes]})
        fidx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()

    if json_path:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)


# CLI

def main():
    ap = argparse.ArgumentParser(description="Photo/Video De-Identifier (faces + text)")
    ap.add_argument("--input", "-i", required=True, help="Image, video, or folder path")
    ap.add_argument("--output", "-o", required=True, help="Output file or folder path")
    ap.add_argument("--mode", choices=["blur", "pixelate", "box"], default="blur", help="Redaction style")
    ap.add_argument("--strength", type=int, default=25, help="Blur/pixelation strength")
    ap.add_argument("--no_text", action="store_true", help="Disable text/plate redaction")
    ap.add_argument("--json", help="(Video) write JSON audit to this path")

    ap.add_argument("--cpu", action="store_true", help="Force CPU (ctx_id=-1, no GPU OCR)")
    ap.add_argument("--det_size", type=int, default=640, help="Face detector input size (e.g., 640, 768, 896)")

    ap.add_argument("--pad_px", type=int, default=6, help="Pad boxes outward in pixels")
    ap.add_argument("--face_thr", type=float, default=0.6, help="Face detection confidence threshold")
    ap.add_argument("--text_thr", type=float, default=0.3, help="Text detection confidence threshold")
    ap.add_argument("--preview", action="store_true", help="For images, save side-by-side preview PNG")

    args = ap.parse_args()

    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
    except Exception:
        pass

    ctx_id = -1 if args.cpu else 0
    include_text = (not args.no_text)

    face_app = build_face_detector(
        ctx_id=ctx_id,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.face_thr
    )
    text_ocr = None if not include_text else build_text_detector(
        use_gpu=(not args.cpu and False),
        det_db_thresh=args.text_thr
    )

    src = args.input
    t0 = time.time()

    if os.path.isdir(src):
        os.makedirs(args.output, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = [os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f.lower())[1] in exts]
        for path_in in tqdm(sorted(imgs), desc="Redacting images", unit="img"):
            base = os.path.splitext(os.path.basename(path_in))[0]
            out_img = os.path.join(args.output, base + "_redacted.jpg")
            try:
                boxes, (W, H) = process_image(
                    path_in, out_img, face_app, text_ocr, args.mode, args.strength,
                    include_text, args.pad_px, args.face_thr, args.text_thr, preview=args.preview
                )
            except Exception as e:
                print(f"[WARN] Failed {path_in}: {e}")
        print(f"Done folder in {time.time() - t0:.1f}s")
        return

    ext = os.path.splitext(src.lower())[1]
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        boxes, (W, H) = process_image(
            src, args.output, face_app, text_ocr, args.mode, args.strength,
            include_text, args.pad_px, args.face_thr, args.text_thr, preview=args.preview
        )
        print(f"Saved redacted image → {args.output}  ({W}x{H})  regions: {len(boxes)}")
        if args.preview:
            print(f"Saved preview → {os.path.splitext(args.output)[0]}_preview.png")
    else:
        process_video(
            src, args.output, face_app, text_ocr, args.mode, args.strength,
            include_text, args.pad_px, args.face_thr, args.text_thr, json_path=args.json
        )
        print(f"Saved redacted video → {args.output}")
        if args.json:
            print(f"Wrote audit JSON → {args.json}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
