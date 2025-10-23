#!/usr/bin/env python3
# deidentify.py
# Photo/Video de-identifier: blur/pixelate/box faces

import os, sys, json, time, argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

# Utilities

def read_image_with_exif(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def save_preview_side_by_side(orig_bgr, red_bgr, out_path_png):
    h = max(orig_bgr.shape[0], red_bgr.shape[0])
    w = orig_bgr.shape[1] + red_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:orig_bgr.shape[0], :orig_bgr.shape[1]] = orig_bgr
    canvas[:red_bgr.shape[0], orig_bgr.shape[1]:] = red_bgr
    cv2.imwrite(out_path_png, canvas)

def to_int_box(xyxy, W, H):
    x1, y1, x2, y2 = map(int, xyxy)
    return [max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)]

def pad_box(box, W, H, pad_px):
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - pad_px),
        max(0, y1 - pad_px),
        min(W-1, x2 + pad_px),
        min(H-1, y2 + pad_px),
    ]

def apply_redaction(img, box, mode, strength):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return
    if mode == "blur":
        k = max(3, strength | 1)
        red = cv2.GaussianBlur(roi, (k, k), 0)
    elif mode == "pixelate":
        h, w = roi.shape[:2]
        f = max(4, strength // 2)
        small = cv2.resize(roi, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_LINEAR)
        red = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:  # "blackbox"
        red = np.zeros_like(roi)
    img[y1:y2, x1:x2] = red

def merge_boxes(boxes, iou_thr=0.2):
    if not boxes: return []
    b = np.array(boxes, dtype=float)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
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


# Detector

def build_face_detector(ctx_id, det_size, det_thresh):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    return app

# Redaction

def redact_frame_faces(frame, face_app, mode, strength, pad_px, face_thr):
    H, W = frame.shape[:2]
    boxes = []
    faces = face_app.get(frame)
    for f in faces:
        score = getattr(f, "det_score", 1.0)
        if score >= face_thr:
            boxes.append(pad_box(to_int_box(f.bbox, W, H), W, H, pad_px))
    boxes = merge_boxes(boxes, iou_thr=0.2)
    for b in boxes:
        apply_redaction(frame, b, mode, strength)
    return frame, boxes

def process_image(path_in, path_out, face_app, mode, strength,
                  pad_px, face_thr, preview):
    orig = read_image_with_exif(path_in)
    img = orig.copy()
    red, boxes = redact_frame_faces(img, face_app, mode, strength, pad_px, face_thr)
    cv2.imwrite(path_out, red)
    if preview:
        prev = os.path.splitext(path_out)[0] + "_preview.png"
        save_preview_side_by_side(orig, red, prev)
    return boxes, (orig.shape[1], orig.shape[0])

def process_video(path_in, path_out, face_app, mode, strength,
                  pad_px, face_thr, json_path):
    cap = cv2.VideoCapture(path_in)
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
        if not ret: break
        frame, boxes = redact_frame_faces(frame, face_app, mode, strength, pad_px, face_thr)
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
    ap = argparse.ArgumentParser(description="Face De-Identifier")
    ap.add_argument("--input", "-i", required=True, help="Image, video, or folder path")
    ap.add_argument("--output", "-o", required=True, help="Output file or folder path")
    ap.add_argument("--mode", choices=["blur","pixelate","box"], default="blur")
    ap.add_argument("--strength", type=int, default=25)
    ap.add_argument("--json", help="(Video) write JSON audit to this path")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for face detector (ctx_id=-1)")
    ap.add_argument("--det_size", type=int, default=640)
    ap.add_argument("--pad_px", type=int, default=6)
    ap.add_argument("--face_thr", type=float, default=0.6)
    ap.add_argument("--preview", action="store_true")

    args = ap.parse_args()

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    ctx_id = -1 if args.cpu else 0
    face_app = build_face_detector(
        ctx_id=ctx_id,
        det_size=(args.det_size, args.det_size),
        det_thresh=args.face_thr
    )

    src = args.input

    if os.path.isdir(src):
        os.makedirs(args.output, exist_ok=True)
        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        imgs = [os.path.join(src,f) for f in os.listdir(src) if os.path.splitext(f.lower())[1] in exts]
        for path_in in tqdm(sorted(imgs), desc="Redacting images", unit="img"):
            base = os.path.splitext(os.path.basename(path_in))[0]
            out_img = os.path.join(args.output, base + "_redacted.jpg")
            process_image(path_in, out_img, face_app, args.mode, args.strength,
                          args.pad_px, args.face_thr, preview=args.preview)
        return

    ext = os.path.splitext(src.lower())[1]
    if ext in [".jpg",".jpeg",".png",".bmp",".webp"]:
        boxes, (W,H) = process_image(src, args.output, face_app, args.mode, args.strength,
                                     args.pad_px, args.face_thr, preview=args.preview)
        print(f"Saved redacted image → {args.output}  ({W}x{H})  faces: {len(boxes)}")
        if args.preview:
            print(f"Saved preview → {os.path.splitext(args.output)[0]}_preview.png")
    else:
        process_video(src, args.output, face_app, args.mode, args.strength,
                      args.pad_px, args.face_thr, json_path=args.json)
        print(f"Saved redacted video → {args.output}")
        if args.json:
            print(f"Wrote audit JSON → {args.json}")

if __name__ == "__main__":
    main()
