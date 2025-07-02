from pathlib import Path
import cv2
import numpy as np
from mathematical_morphology import preprocess_image, detect_sunspots_morphology, mask_to_bboxes

def load_yolo_labels(label_file, img_shape):
    bboxes = []
    h, w = img_shape
    with open(label_file, 'r') as f:
        for line in f:
            _, x, y, bw, bh = map(float, line.strip().split())
            x_min = int((x - bw/2) * w)
            y_min = int((y - bh/2) * h)
            x_max = int((x + bw/2) * w)
            y_max = int((y + bh/2) * h)
            bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea) if (box1Area + box2Area - interArea) > 0 else 0
    return iou

def match_bboxes(pred_bboxes, gt_bboxes, iou_thr):
    matched_gt = set()
    tp = 0
    for pred in pred_bboxes:
        best_iou = 0
        best_gt = -1
        for i, gt in enumerate(gt_bboxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gt)
    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes) - tp
    return tp, fp, fn

def main():
    val_images_path = Path('dataset/OGAUC/valid/images')
    val_labels_path = Path('dataset/OGAUC/valid/labels')

    total_tp, total_fp, total_fn = 0, 0, 0
    all_precisions_50 = []
    all_precisions_95 = []

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []

    for img_file in val_images_path.glob('*.jpg'):
        label_file = val_labels_path / (img_file.stem + '.txt')
        if not label_file.exists():
            continue

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        preprocessed, _ = preprocess_image(str(img_file))
        _, mask = detect_sunspots_morphology(preprocessed)
        pred_bboxes = mask_to_bboxes(mask,min_area=2)
        gt_bboxes = load_yolo_labels(label_file, (h, w))

        # Para precisión y recall estándar (IoU 0.5)
        tp, fp, fn = match_bboxes(pred_bboxes, gt_bboxes, iou_thr=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Para mAP50-95
        precisions = []
        for thr in iou_thresholds:
            tp_thr, fp_thr, fn_thr = match_bboxes(pred_bboxes, gt_bboxes, iou_thr=thr)
            precision_thr = tp_thr / (tp_thr + fp_thr) if (tp_thr + fp_thr) > 0 else 0
            precisions.append(precision_thr)
        aps.append(np.mean(precisions))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    mAP50 = np.mean([ap for ap in aps]) if aps else 0
    mAP5095 = np.mean(aps) if aps else 0

    print(f'Precisión: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'mAP50: {mAP50:.4f}')
    print(f'mAP50-95: {mAP5095:.4f}')

if __name__ == "__main__":
    main()