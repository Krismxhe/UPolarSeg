"""
Generate full-size masks + extract PPA biomarkers.
"""

import argparse
from pathlib import Path
import numpy as np, cv2, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.seg_module import SegModule
from src.models.outputs import normalize_model_output
from hydra import compose, initialize_config_dir

CENTER, RADIUS = (256, 256), 362

def postprocess(m, ks=7, ma=20):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mc = m.copy()
    bod = (mc == 1).astype(np.uint8)
    if bod.sum() > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bod, 8)
        if n > 1: bod = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        bod = cv2.morphologyEx(bod, cv2.MORPH_OPEN, k); bod = cv2.morphologyEx(bod, cv2.MORPH_CLOSE, k)
        mc[bod == 1] = 1; mc[(bod == 0) & (mc == 1)] = 0
    bpp = (mc == 2).astype(np.uint8)
    if bpp.sum() > 0:
        bpp = cv2.morphologyEx(bpp, cv2.MORPH_OPEN, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bpp, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < ma: bpp[lab == i] = 0
        mc[bpp == 1] = 2; mc[(bpp == 0) & (mc == 2)] = 0
    return mc

def measure(pm):
    od = pm == 1; ppa = pm == 2
    spans = np.sum(od, axis=1).astype(float)
    if spans.sum() == 0: return 0, 0, 0
    H = len(spans); theta_max = np.argmax(spans)
    v = spans[theta_max] + spans[(theta_max + H//2) % H]
    h = spans[(theta_max + H//4) % H] + spans[(theta_max + 3*H//4) % H]
    pw = max((np.sum(ppa[r, int(s):min(int(s)+80, 512)]) if (s:=spans[r]) > 0 else 0) for r in range(H))
    return v, h, pw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True, help='Directory of passed images (stem.png)')
    parser.add_argument('--crop_params', required=True, help='crop_params_all.csv')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ckpt', required=True, help='UPolarSeg checkpoint')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.output_dir) / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose('train', overrides=['model=research/modular_unet_pmsr_lite', 'polar=enabled', 'dataset=beijing_1024_polar', 'train.img_size=512'])
    model = SegModule.load_from_checkpoint(args.ckpt, cfg=cfg, map_location='cpu').eval().cuda(args.gpu)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(args.gpu)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(args.gpu)

    params = pd.read_csv(args.crop_params)
    results = []

    for _, row in tqdm(params.iterrows(), total=len(params), desc='Processing'):
        stem = row['stem']
        img_path = img_dir / f'{stem}.png'
        if not img_path.exists(): continue

        img = np.array(Image.open(img_path).convert('RGB'))

        x1c, y1c = int(row['x1c']), int(row['y1c'])
        x2c, y2c = int(row['x2c']), int(row['y2c'])
        pad_l, pad_t = int(row['pad_left']), int(row['pad_top'])
        pad_r, pad_b = int(row['pad_right']), int(row['pad_bottom'])

        crop = img[y1c:y2c, x1c:x2c]
        if pad_l or pad_t or pad_r or pad_b:
            crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')

        img_512 = np.array(Image.fromarray(crop).resize((512, 512), Image.BILINEAR))
        img_polar = cv2.warpPolar(img_512.astype(np.float32), (512, 512), CENTER, RADIUS,
                                  cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)

        tensor = torch.from_numpy(img_polar).permute(2, 0, 1).unsqueeze(0).float().cuda(args.gpu) / 255.0
        tensor = (tensor - mean) / std

        with torch.no_grad():
            output = model(tensor)
            logits = normalize_model_output(output)['seg_logits']
            pred_polar = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_pp = postprocess(pred_polar)
        v, h, pw = measure(pred_pp)
        results.append({'stem': stem, 'OD_vertical': v, 'OD_horizontal': h, 'PPA_max_width': pw,
                        'HVDR': h/v if v>0 else 0, 'PVDR': pw/v if v>0 else 0})

        pred_cart = cv2.warpPolar(pred_polar.astype(np.float32), (512, 512), CENTER, RADIUS,
                                  cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST)
        pred_cart = np.clip(np.round(pred_cart), 0, 2).astype(np.uint8)
        pred_1024 = np.array(Image.fromarray(pred_cart).resize((1024, 1024), Image.NEAREST))

        h_orig, w_orig = int(row['h_orig']), int(row['w_orig'])
        full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        pred_unpad = pred_1024[pad_t:1024 - pad_b, pad_l:1024 - pad_r]
        full_mask[y1c:y2c, x1c:x2c] = pred_unpad
        Image.fromarray(full_mask).save(mask_dir / f'{stem}.png')

    df = pd.DataFrame(results)
    df.to_csv(Path(args.output_dir) / 'biomarkers.csv', index=False)
    print(f'Done: {len(df)} masks + biomarkers saved to {args.output_dir}')
    print(df.describe())

if __name__ == '__main__':
    main()
