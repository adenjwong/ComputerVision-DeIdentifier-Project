# ComputerVision-DeIdentifier-Project

### Blur, pixelate, or black-box faces in photos and videos — powered by [InsightFace (RetinaFace)](https://github.com/deepinsight/insightface) and OpenCV.

---

## Overview

This tool automatically detects faces in images or videos and redacts them by **blurring**, **pixelating**, or **boxing**.  
It’s designed to anonymize people in visual data quickly and locally — no cloud APIs required.

---

## Features

- Detects all visible faces in images or videos  
- Choose between **blur**, **pixelate**, or **black box** modes  
- Adjustable intensity (`--strength`) and padding (`--pad_px`)  
- Outputs JSON with bounding boxes for videos (optional)  
- Lightweight: runs on Python 3.10 with no CUDA needed  


---

## Set-up

### Create environment
```bash
conda create -n deid python=3.10 -y
conda activate deid
```

### install dependencies
```bash
pip install \
  "numpy==1.26.4" \
  "opencv-python==4.9.0.80" \
  "onnxruntime==1.18.0" \
  "insightface==0.7.3" \
  pillow==10.4.0 tqdm==4.67.1
```

## Usage

### anonymize a single photo
```bash
python deidentify_faces.py -i partialface.jpg -o blurpartialface.jpg --cpu --preview \
  --mode blur --strength 40 --pad_px 8 --det_size 768
```
**Example Output:**  
`output.jpg` → blurred faces  
`output_preview.png` → side-by-side comparison  

![black-box multiple faces image test](Test/boxmultiface_preview.png)

### anonymize all faces in a video
```bash
Copy code
python deidentify_faces.py -i inputvideo.mp4 -o boxedvideo.mp4 --cpu \
  --mode box --pad_px 10 --det_size 768 --face_thr 0.6 --json boxed_faces.json
```

**Example Output:**  
`output.mp4` → redacted video  
`output_faces.json` → frame-by-frame face bounding boxes  

![black-box multiple faces video example](Test/boxedvideo_preview.gif)

## Command-Line Arguments

| Flag | Description |
|------|--------------|
| `-i, --input` | Path to image, video, or folder of images |
| `-o, --output` | Output path (file or folder) |
| `--mode` | Redaction type: `blur`, `pixelate`, or `box` |
| `--strength` | Intensity of blur/pixelation (ignored for `box`) |
| `--pad_px` | Expand detection boxes by this many pixels |
| `--det_size` | Face detector resolution (640–896 recommended) |
| `--face_thr` | Minimum confidence threshold (0–1) |
| `--cpu` | Force CPU mode (recommended on macOS) |
| `--preview` | Save side-by-side comparison image (for stills) |
| `--json` | Output JSON with detection boxes (videos only) |


---

## Credits
- Face Detection: InsightFace / RetinaFace
- Image Processing: OpenCV
- Pillow: EXIF-aware image loading

---

