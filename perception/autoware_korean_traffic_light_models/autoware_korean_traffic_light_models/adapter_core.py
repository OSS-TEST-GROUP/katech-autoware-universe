"""Model-specific preprocessing and fail-closed experimental signal mapping.

Thresholds are provisional engineering gates, not validated safety guarantees.
Six-output candidate C; unsupported states can still score highly. Test-only.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image

LABELS = ("red", "yellow", "green", "greenleft", "redleft")
ELEMENTS = (
    ((1, 5),), ((2, 5),), ((3, 5),), ((3, 5), (3, 6)), ((1, 5), (3, 6))
)


@dataclass(frozen=True)
class ROI:
    identifier: int
    kind: int
    x: int
    y: int
    width: int
    height: int


def unknown(roi, reason):
    return {"id": roi.identifier, "type": roi.kind, "label":"unknown", "reason": reason,
            "confidence":0.0, "elements":[{"color":18, "shape":18, "status":18, "confidence":0.0}]}


def decode(probabilities, roi, min_confidence=0.8, min_margin=0.2):
    p = np.asarray(probabilities)
    if p.shape != (6,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1.0001) or abs(float(p.sum())-1)>1e-3:
        return unknown(roi, "invalid_probabilities")
    indices = np.argsort(p)
    best = int(indices[-1])
    if best == 5:
        return unknown(roi, "learned_reject")
    confidence = float(p[best])
    if confidence < min_confidence or confidence-float(p[indices[-2]]) < min_margin:
        return unknown(roi, "low_confidence_or_margin")
    return {"id":roi.identifier,"type":roi.kind,"label":LABELS[best],"reason":"classified",
            "confidence":confidence, "elements":[{"color":color,"shape":shape,"status":16,
            "confidence":confidence} for color,shape in ELEMENTS[best]]}


def preprocess(crop):
    if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError("Expected nonempty uint8 RGB crop")
    # Match the original dataset loader, not Autoware's ImageNet/letterbox default.
    image = Image.fromarray(crop).resize((70,20), resample=Image.Resampling.BICUBIC)
    return np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()/255.0


def image_from_message(message):
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError("unsupported_encoding")
    h,w,step = int(message.height),int(message.width),int(message.step)
    if min(h,w) <= 0 or max(h,w)>4096 or step < w*3 or step*h > 64*1024**2:
        raise ValueError("invalid_image_dimensions")
    data = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if len(data) != h*step:
        raise ValueError("invalid_image_buffer")
    image = data.reshape(h,step)[:,:w*3].reshape(h,w,3)
    if message.encoding == "bgr8":
        image = image[:,:,::-1]
    return np.ascontiguousarray(image)


def classify(image, rois, backend, min_confidence=0.8, min_margin=0.2):
    if not 0 <= min_confidence <= 1 or not 0 <= min_margin <= 1:
        raise ValueError("Invalid confidence settings")
    if len(rois)>64:
        return [unknown(r,"too_many_rois") for r in rois[:64]]
    if len({r.identifier for r in rois}) != len(rois):
        return [unknown(r,"duplicate_ids") for r in dict((r.identifier,r) for r in rois).values()]
    results = [None]*len(rois)
    tensors, positions = [],[]
    h,w = image.shape[:2]
    for i,r in enumerate(rois):
        if r.kind != 0:
            results[i] = unknown(r,"unsupported_traffic_light_type")
        elif min(r.width,r.height)<4 or r.x<0 or r.y<0 or r.x+r.width>w or r.y+r.height>h:
            results[i] = unknown(r,"invalid_or_tiny_roi")
        else:
            crop = image[r.y:r.y+r.height,r.x:r.x+r.width]
            if float(np.std(crop.astype(np.float32)))<1.0:
                results[i] = unknown(r,"uniform_crop")
            else:
                tensors.append(preprocess(crop))
                positions.append(i)
    try:
        for start in range(0,len(tensors),backend.batch):
            current = tensors[start:start+backend.batch]
            count = len(current)
            data = np.stack(current+[current[-1]]*(backend.batch-count))
            output = backend.infer(data)
            if output.shape != (backend.batch,6):
                raise ValueError("Invalid backend output shape")
            for j in range(count):
                i = positions[start+j]
                results[i] = decode(output[j],rois[i],min_confidence,min_margin)
    except Exception:
        for i in positions:
            results[i] = unknown(rois[i],"inference_error")
    return results

