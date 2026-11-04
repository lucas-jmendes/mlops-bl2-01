"""Novas características de imagem para monitoramento do servidor BentoML.

Complementa o image_extract.py da aula 011 (médias dos canais L, R, G e B)
com três características:

1. sharpness     (numérica)   variância do Laplaciano do canal L
2. colorfulness  (numérica)   métrica de Hasler & Süsstrunk (2003), eixos de crominância
3. main_object   (categórica) objeto principal detectado pela rede YOLO11n (COCO, 80 classes),
                              com main_object_conf e n_objects como apoio

Uso:
    python extract_features.py imagens/*.webp

Saídas:
    resultados.json, resultados.csv e figuras/deteccoes_<imagem>.jpg
"""

from __future__ import annotations

import ast
import csv
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

ROOT = Path(__file__).parent
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
MODEL_PATH = ROOT / "models" / "yolo11n.onnx"
YOLO_SIZE = 640        # tamanho de entrada da rede
CONF_THRESHOLD = 0.50  # confiança mínima para aceitar uma detecção
IOU_THRESHOLD = 0.45   # sobreposição máxima no NMS


# ------------------------------------------------- baseline da aula (L, R, G, B)
def channel_mean(image: PILImage, channel: str) -> float:
    """Média do canal pelo histograma, igual ao luminance_mean da aula 011."""
    band = image.convert("L") if channel == "L" else image.convert("RGB").getchannel(channel)
    counts = band.histogram()
    return sum(i * c for i, c in enumerate(counts)) / sum(counts)


# ------------------------------------------------------------- 1. nitidez
def sharpness(image: PILImage) -> float:
    """Variância do Laplaciano no canal L.

    O Laplaciano é uma segunda derivada e responde forte nas bordas.
    Imagem nítida e com textura -> variância alta.
    Imagem borrada, com neblina, muito comprimida ou sem textura -> variância baixa.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ------------------------------------------------------------ 2. colorido
def colorfulness(image: PILImage) -> float:
    """Métrica de Hasler & Süsstrunk.

    rg = R - G             (eixo vermelho-verde)
    yb = (R + G)/2 - B     (eixo amarelo-azul)
    C  = sqrt(std_rg² + std_yb²) + 0.3 * sqrt(mean_rg² + mean_yb²)

    rg e yb são eixos oponentes de cor (mesma ideia da crominância):
    separam a cor do brilho. Uma imagem em tons de cinza dá C = 0.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


# ------------------------------------------------------ 3. objeto (IA)
def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Baixa o YOLO11n em ONNX (≈10 MB) na primeira execução."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando modelo de {MODEL_URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class ObjectDetector:
    """Detector YOLO11n (Ultralytics) exportado em ONNX, rodando no onnxruntime (CPU)."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.session = ort.InferenceSession(str(ensure_model(model_path)),
                                            providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta["names"])  # {0: 'person', 1: 'bicycle', ...}

    def _letterbox(self, rgb: np.ndarray):
        """Redimensiona mantendo a proporção e completa com cinza até 640x640."""
        h, w = rgb.shape[:2]
        scale = YOLO_SIZE / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
        top, left = (YOLO_SIZE - nh) // 2, (YOLO_SIZE - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, (left, top)

    def detect(self, image: PILImage) -> list[dict]:
        rgb = np.asarray(image.convert("RGB"))
        canvas, scale, (left, top) = self._letterbox(rgb)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {"images": blob})[0][0].T  # (8400, 4 + 80)

        boxes_xywh, scores = out[:, :4], out[:, 4:]
        class_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        keep = confs >= CONF_THRESHOLD
        boxes_xywh, class_ids, confs = boxes_xywh[keep], class_ids[keep], confs[keep]

        # caixa (centro x, centro y, larg, alt) no canvas -> (x, y, larg, alt) na imagem original
        boxes = []
        for cx, cy, bw, bh in boxes_xywh:
            boxes.append([float((cx - bw / 2 - left) / scale), float((cy - bh / 2 - top) / scale),
                          float(bw / scale), float(bh / scale)])

        idx = cv2.dnn.NMSBoxes(boxes, confs.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)
        area = rgb.shape[0] * rgb.shape[1]
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = boxes[i]
            dets.append({
                "label": self.names[int(class_ids[i])],
                "conf": round(float(confs[i]), 3),
                "area_frac": round(bw * bh / area, 4),
                "box": [round(v, 1) for v in (x, y, bw, bh)],
            })
        return dets


def main_object(dets: list[dict]) -> tuple[str, float]:
    """Objeto principal = detecção com maior área × confiança; 'nenhum' se não houver."""
    if not dets:
        return "nenhum", 0.0
    best = max(dets, key=lambda d: d["area_frac"] * d["conf"])
    return best["label"], best["conf"]


# ------------------------------------------------------------ junta tudo
def extract_extra_features(image: PILImage, detector: ObjectDetector) -> dict:
    dets = detector.detect(image)
    label, conf = main_object(dets)
    return {
        "luminance_mean": round(channel_mean(image, "L"), 2),
        "r_mean": round(channel_mean(image, "R"), 2),
        "g_mean": round(channel_mean(image, "G"), 2),
        "b_mean": round(channel_mean(image, "B"), 2),
        "sharpness": round(sharpness(image), 2),
        "colorfulness": round(colorfulness(image), 2),
        "main_object": label,
        "main_object_conf": conf,
        "n_objects": len(dets),
        "detections": dets,
    }


def draw_detections(image: PILImage, dets: list[dict], out: Path) -> None:
    img = image.convert("RGB")
    s = img.width / 1000
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, int(20 * s)))
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    for d in dets:
        x, y, w, h = d["box"]
        draw.rectangle([x, y, x + w, y + h], outline=(230, 120, 20), width=max(3, int(4 * s)))
        text = f"{d['label']} {d['conf']:.2f}"
        th = getattr(font, "size", 10) + 6
        draw.rectangle([x, y - th, x + draw.textlength(text, font=font) + 8, y], fill=(230, 120, 20))
        draw.text((x + 4, y - th + 2), text, fill="white", font=font)
    img.thumbnail((1000, 1000))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)


def main(paths: list[str]) -> None:
    detector = ObjectDetector()
    results = {}
    for arg in paths:
        path = Path(arg)
        with Image.open(path) as img:
            img.load()
            feats = extract_extra_features(img, detector)
            draw_detections(img, feats["detections"], ROOT / "figuras" / f"deteccoes_{path.stem}.jpg")
        results[path.name] = feats

    (ROOT / "resultados.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    cols = ["luminance_mean", "r_mean", "g_mean", "b_mean", "sharpness", "colorfulness",
            "main_object", "main_object_conf", "n_objects"]
    with open(ROOT / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", *cols])
        for name, feats in results.items():
            writer.writerow([name, *(feats[c] for c in cols)])

    print(json.dumps({k: {c: v[c] for c in cols} for k, v in results.items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: python extract_features.py imagem1 [imagem2 ...]")
    main(sys.argv[1:])
