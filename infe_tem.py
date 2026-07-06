"""Morphology inference"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import torch
from omegaconf import DictConfig
import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PPA_MARGIN = 65

def normalize_mask(m):
    if m.max() > 2:
        out = np.zeros_like(m, dtype=np.uint8); out[m==127]=1; out[m==255]=2
        return out
    return m

def mask_to_visual(m):
    vis = np.zeros_like(m, dtype=np.uint8); vis[m==1]=127; vis[m==2]=255
    return vis

def postprocess_polar_mask(mask, kernel_size=7, min_area=20):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = mask.astype(np.uint8).copy()
    binary_od = (mask_clean == 1).astype(np.uint8)
    if binary_od.sum() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_od, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_od = (labels == largest_label).astype(np.uint8)
        binary_od = cv2.morphologyEx(binary_od, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_od = cv2.morphologyEx(binary_od, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_clean[binary_od == 1] = 1
        mask_clean[(binary_od == 0) & (mask_clean == 1)] = 0
    binary_ppa = (mask_clean == 2).astype(np.uint8)
    if binary_ppa.sum() > 0:
        binary_ppa = cv2.morphologyEx(binary_ppa, cv2.MORPH_OPEN, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_ppa, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary_ppa[labels == i] = 0
        mask_clean[binary_ppa == 1] = 2
        mask_clean[(binary_ppa == 0) & (mask_clean == 2)] = 0
    return mask_clean

def measure_polar_mask(polar_mask, ppa_margin=PPA_MARGIN):
    polar_mask = normalize_mask(polar_mask)
    od_mask = polar_mask == 1; ppa_mask = polar_mask == 2
    od_spans = np.sum(od_mask, axis=1)
    if od_spans.sum() == 0: return 0, 0, 0
    valid_od = od_spans[od_spans > 0]
    od_vertical = np.max(valid_od); od_horizontal = np.min(valid_od)
    ppa_max_width = 0
    for r in range(polar_mask.shape[0]):
        od_span = od_spans[r]
        if od_span == 0: continue
        search_end = min(od_span + ppa_margin, polar_mask.shape[1])
        ppa_count = np.sum(ppa_mask[r, od_span:search_end])
        ppa_max_width = max(ppa_max_width, ppa_count)
    return od_vertical, od_horizontal, ppa_max_width

def save_error_heatmap(pred, gt, save_path):
    h, w = pred.shape
    fp = np.zeros((h,w,3), dtype=np.uint8); fn = np.zeros((h,w,3), dtype=np.uint8)
    for c in range(1,3):
        pc=(pred==c); gc=(gt==c)
        fp[pc&~gc]=[255,0,0]; fn[gc&~pc]=[0,0,255]
    Image.fromarray(cv2.addWeighted(fp,0.5,fn,0.5,0)).save(save_path)

def save_prob_heatmap(prob, save_path):
    plt.imsave(save_path, np.maximum(prob[1], prob[2]), cmap='jet', vmin=0, vmax=1)

@hydra.main(version_base=None, config_path='configs', config_name='train')
def main(cfg: DictConfig):
    from src.models.seg_module import SegModule
    from src.models.outputs import normalize_model_output
    from src.datasets.seg_dataset import SegDataModule
    from src.datasets.batch import unpack_batch
    from hydra.utils import to_absolute_path
    
    ckpt_path = cfg.get('checkpoint', None)
    if ckpt_path is None:
        raise ValueError("Please provide checkpoint=outputs/.../best.ckpt")
    
    output_dir_str = cfg.get('output_dir', 'predictions/morphology')
    kernel_size = cfg.get('kernel_size', 7)
    min_area = cfg.get('min_area', 20)
    
    output_dir = Path(to_absolute_path(output_dir_str))
    dirs = {
        'mask_polar': output_dir/'mask_polar', 'prob_polar': output_dir/'prob_polar',
        'error_polar': output_dir/'error_polar', 'postproc': output_dir/'postprocessed_polar',
        'metrics': output_dir/'metrics',
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    datamodule = SegDataModule(cfg, eval_split='test')
    datamodule.setup(stage='test')
    model = SegModule.load_from_checkpoint(ckpt_path, cfg=cfg).eval().cuda()

    num_classes = cfg.dataset.num_classes
    tp = np.zeros(num_classes, dtype=np.float64); fp = np.zeros(num_classes, dtype=np.float64); fn = np.zeros(num_classes, dtype=np.float64)
    
    # ── Per-grade accumulators ──
    grade_tp = {g: np.zeros(num_classes, dtype=np.float64) for g in ['high_myopia','moderate_myopia','low_myopia']}
    grade_fp = {g: np.zeros(num_classes, dtype=np.float64) for g in ['high_myopia','moderate_myopia','low_myopia']}
    grade_fn = {g: np.zeros(num_classes, dtype=np.float64) for g in ['high_myopia','moderate_myopia','low_myopia']}

    morph_records = []
    od_total = ppa_total = bg_total = 0
    od_to_bg = od_to_ppa = ppa_to_bg = ppa_to_od = bg_to_od = bg_to_ppa = 0

    loader = datamodule.test_dataloader()

    for batch in tqdm(loader, desc='Processing'):
        images, masks, meta = unpack_batch(batch)
        with torch.no_grad():
            output = model(images.cuda())
            logits = normalize_model_output(output)['seg_logits']
            prob = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        for b in range(preds.shape[0]):
            pred_polar = preds[b].cpu().numpy().astype(np.uint8)
            gt_polar = masks[b].cpu().numpy().astype(np.uint8)
            prob_np = prob[b].cpu().numpy()
            
            case_id = meta.get('case_id', [f'{b}'])[b] if isinstance(meta.get('case_id'), list) else str(b)
            stem = case_id if isinstance(case_id, str) else str(case_id)

            # Dice/IoU
            for c in range(num_classes):
                pc=(pred_polar==c); gc=(gt_polar==c)
                tp[c]+=(pc&gc).sum(); fp[c]+=(pc&~gc).sum(); fn[c]+=(~pc&gc).sum()

            # ── Per-grade Dice/IoU ──
            gr = next((g for g in ['high_myopia','moderate_myopia','low_myopia'] if stem.startswith(g)), None)
            if gr:
                for c in range(num_classes):
                    pc=(pred_polar==c); gc=(gt_polar==c)
                    grade_tp[gr][c] += (pc&gc).sum(); grade_fp[gr][c] += (pc&~gc).sum(); grade_fn[gr][c] += (~pc&gc).sum()

            # postprocess + morphology
            pred_polar_pp = postprocess_polar_mask(pred_polar, kernel_size=kernel_size, min_area=min_area)
            gt_vert, gt_horiz, gt_ppa = measure_polar_mask(gt_polar)
            pred_vert, pred_horiz, pred_ppa = measure_polar_mask(pred_polar_pp)
            hvdr_gt = gt_horiz/gt_vert if gt_vert>0 else 0; pvdr_gt = gt_ppa/(2*gt_vert) if gt_vert>0 else 0
            hvdr_pred = pred_horiz/pred_vert if pred_vert>0 else 0; pvdr_pred = pred_ppa/(2*pred_vert) if pred_vert>0 else 0
            morph_records.append({
                'stem': stem, 'gt_OD_vertical': gt_vert, 'gt_OD_horizontal': gt_horiz, 'gt_PPA_max_width': gt_ppa,
                'pred_OD_vertical': pred_vert, 'pred_OD_horizontal': pred_horiz, 'pred_PPA_max_width': pred_ppa,
                'gt_HVDR': hvdr_gt, 'gt_PVDR': pvdr_gt, 'pred_HVDR': hvdr_pred, 'pred_PVDR': pvdr_pred,
            })

            # Confusion
            od_total += (gt_polar==1).sum(); ppa_total += (gt_polar==2).sum(); bg_total += (gt_polar==0).sum()
            od_to_bg += ((gt_polar==1)&(pred_polar==0)).sum(); od_to_ppa += ((gt_polar==1)&(pred_polar==2)).sum()
            ppa_to_bg += ((gt_polar==2)&(pred_polar==0)).sum(); ppa_to_od += ((gt_polar==2)&(pred_polar==1)).sum()
            bg_to_od += ((gt_polar==0)&(pred_polar==1)).sum(); bg_to_ppa += ((gt_polar==0)&(pred_polar==2)).sum()

            Image.fromarray(mask_to_visual(pred_polar)).save(dirs['mask_polar']/f"{stem}.png")
            Image.fromarray(mask_to_visual(pred_polar_pp)).save(dirs['postproc']/f"{stem}.png")
            np.savez_compressed(dirs['prob_polar']/f"{stem}.npz", prob=prob_np)
            save_prob_heatmap(prob_np, dirs['prob_polar']/f"{stem}.png")
            save_error_heatmap(pred_polar, gt_polar, dirs['error_polar']/f"{stem}.png")

    # summary
    dice = 2*tp/(2*tp+fp+fn+1e-8); iou = tp/(tp+fp+fn+1e-8)
    class_names = cfg.dataset.class_names; fg_ids = cfg.dataset.foreground_classes
    print(f"\nSamples: {len(morph_records)}")
    print("\nDice/IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:>12}: Dice={dice[i]:.4f}, IoU={iou[i]:.4f}")
    print(f"  {'mean(fg)':>12}: Dice={dice[fg_ids].mean():.4f}, IoU={iou[fg_ids].mean():.4f}")

    df = pd.DataFrame(morph_records)
    df.to_csv(dirs['metrics']/'morphology_per_case.csv', index=False)
    print("\nMorphology MAE/RMSE:")
    for col in ['OD_vertical','OD_horizontal','PPA_max_width','HVDR','PVDR']:
        gt_c=f'gt_{col}'; pred_c=f'pred_{col}'
        mae=mean_absolute_error(df[gt_c],df[pred_c])
        rmse=np.sqrt(mean_squared_error(df[gt_c],df[pred_c]))
        print(f"  {col:>18}: MAE={mae:>8.4f}  RMSE={rmse:>8.4f}")

    print("\nConfusion:")
    print(f"  OD→BG  = {od_to_bg/od_total*100:.2f}%")
    print(f"  OD→PPA = {od_to_ppa/od_total*100:.2f}%")
    print(f"  OD FN  = {(od_to_bg+od_to_ppa)/od_total*100:.2f}%")
    print(f"  PPA→BG = {ppa_to_bg/ppa_total*100:.2f}%")
    print(f"  PPA→OD = {ppa_to_od/ppa_total*100:.2f}%")
    print(f"  PPA FN = {(ppa_to_bg+ppa_to_od)/ppa_total*100:.2f}%")
    print(f"  BG→OD  = {bg_to_od/bg_total*100:.2f}%")
    print(f"  BG→PPA = {bg_to_ppa/bg_total*100:.2f}%")

    summary = {
        'metric': ['OD_vertical','OD_horizontal','PPA_max_width','HVDR','PVDR',
                   'OD->BG','OD->PPA','OD_FN','PPA->BG','PPA->OD','PPA_FN','BG->OD','BG->PPA',
                   'Dice_OD','Dice_PPA','IoU_OD','IoU_PPA'],
        'value': [
            mean_absolute_error(df['gt_OD_vertical'],df['pred_OD_vertical']),
            mean_absolute_error(df['gt_OD_horizontal'],df['pred_OD_horizontal']),
            mean_absolute_error(df['gt_PPA_max_width'],df['pred_PPA_max_width']),
            mean_absolute_error(df['gt_HVDR'],df['pred_HVDR']),
            mean_absolute_error(df['gt_PVDR'],df['pred_PVDR']),
            od_to_bg/od_total*100, od_to_ppa/od_total*100, (od_to_bg+od_to_ppa)/od_total*100,
            ppa_to_bg/ppa_total*100, ppa_to_od/ppa_total*100, (ppa_to_bg+ppa_to_od)/ppa_total*100,
            bg_to_od/bg_total*100, bg_to_ppa/bg_total*100,
            dice[1].item(), dice[2].item(), iou[1].item(), iou[2].item(),
        ]
    }
    pd.DataFrame(summary).to_csv(dirs['metrics']/'summary.csv', index=False)
    dice_iou = [{'class': class_names[i], 'dice': dice[i].item(), 'iou': iou[i].item()} for i in range(num_classes)]
    pd.DataFrame(dice_iou).to_csv(dirs['metrics']/'dice_iou.csv', index=False)

    # ── Per-grade summary ──
    grade_rows = []
    for gr in ['low_myopia', 'moderate_myopia', 'high_myopia']:
        gdice = 2*grade_tp[gr] / (2*grade_tp[gr] + grade_fp[gr] + grade_fn[gr] + 1e-8)
        sub = df[df['stem'].str.startswith(gr)]
        if len(sub) == 0: continue
        grade_rows.append({
            'grade': gr,
            'Dice_OD': round(gdice[1].item(), 2),
            'Dice_PPA': round(gdice[2].item(), 2),
            'MAE_HVDR': round(mean_absolute_error(sub['gt_HVDR'], sub['pred_HVDR']), 4),
            'MAE_PPA_W': round(mean_absolute_error(sub['gt_PPA_max_width'], sub['pred_PPA_max_width']), 1),
            'MAE_PVDR': round(mean_absolute_error(sub['gt_PVDR'], sub['pred_PVDR']), 4),
        })
    pd.DataFrame(grade_rows).to_csv(dirs['metrics']/'per_grade_summary.csv', index=False)

    print(f"\nAll outputs saved to {output_dir}")

if __name__ == '__main__':
    main()