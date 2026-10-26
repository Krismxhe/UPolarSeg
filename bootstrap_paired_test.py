"""
Patient-level bootstrap paired test for model comparison.

Compares a target model against the best baseline for each metric,
using patient-level bootstrap to account for multiple observations
from the same patient (e.g., bilateral eyes).

Usage:
    python bootstrap_paired_test.py \
        --csvs /path/to/per_case_beijing_all.csv /path/to/per_case_tibet_all.csv \
        --target UPolarSeg \
        --n_boot 1000 \
        --output paired_test_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#   - Dice  tp/fp/fn global Dice
#   - MAE  mean
DEFAULT_TASKS = [
    ('OD Dice',  'OD',       'dice'),
    ('PPA Dice', 'PPA',      'dice'),
    ('HVDR MAE', 'hvdr_err', 'mae'),
    ('PMW MAE',  'pmw_err',  'mae'),
    ('PVDR MAE', 'pvdr_err', 'mae'),
]


def global_dice(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn + 1e-8)


def model_global_dice(df, model, cls):
    sub = df[df['model'] == model]
    tp = sub[f'tp_{cls}'].sum()
    fp = sub[f'fp_{cls}'].sum()
    fn = sub[f'fn_{cls}'].sum()
    return global_dice(tp, fp, fn)


def _expand_by_patient_counts(merged, sampled_patients):
    counts = pd.Series(sampled_patients).value_counts()
    boot = merged.merge(
        counts.rename('w'),
        left_on='patient_id', right_index=True, how='left',
    )
    boot['w'] = boot['w'].fillna(0).astype(int)
    return boot.loc[boot.index.repeat(boot['w'])].drop(columns='w')


def bootstrap_paired(df, target_model, baseline_model,
                     metric_key, direction, metric_type,
                     n_boot=1000, rng=None):
    """
    patient-level Bootstrap paired test

    Returns
    -------
    dict with keys: obs_diff, ci_low, ci_high, p_value,
                    n_patients, n_cases.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df_up = df[df['model'] == target_model]
    df_bl = df[df['model'] == baseline_model]
    merged = df_up.merge(df_bl, on=['case_id', 'patient_id'],
                         suffixes=('_up', '_bl'))

    patients = merged['patient_id'].unique()
    n_patients = len(patients)
    n_cases = len(merged)

    def compute_diff(sub):
        if metric_type == 'dice':
            cls = metric_key
            tp_up = sub[f'tp_{cls}_up'].sum()
            fp_up = sub[f'fp_{cls}_up'].sum()
            fn_up = sub[f'fn_{cls}_up'].sum()
            tp_bl = sub[f'tp_{cls}_bl'].sum()
            fp_bl = sub[f'fp_{cls}_bl'].sum()
            fn_bl = sub[f'fn_{cls}_bl'].sum()
            v_up = global_dice(tp_up, fp_up, fn_up)
            v_bl = global_dice(tp_bl, fp_bl, fn_bl)
        else:
            v_up = sub[f'{metric_key}_up'].mean()
            v_bl = sub[f'{metric_key}_bl'].mean()
        # delta>0, better
        if direction == 'higher_better':
            return v_up - v_bl
        return v_bl - v_up

    obs_diff = compute_diff(merged)

    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled = rng.choice(patients, size=n_patients, replace=True)
        boot = _expand_by_patient_counts(merged, sampled)
        diffs[i] = compute_diff(boot)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs < 0).mean(), (diffs > 0).mean())
    p_value = min(p_value, 1.0)

    return {
        'obs_diff': obs_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value,
        'n_patients': n_patients,
        'n_cases': n_cases,
    }


def run_dataset(df, dataset_name, target_model, tasks, n_boot, rng):
    results = []
    models = [m for m in df['model'].unique() if m != target_model]

    if target_model not in df['model'].unique():
        print(f'[WARN] {dataset_name}: target model "{target_model}" not found, skip.')
        return pd.DataFrame()

    for name, key, metric_type in tasks:
        # choose best baseline
        if metric_type == 'dice':
            scores = {m: model_global_dice(df, m, key) for m in models}
            best = max(scores, key=scores.get)
            direction = 'higher_better'
        else:
            scores = df[df['model'] != target_model].groupby('model')[key].mean()
            best = scores.idxmin()
            direction = 'lower_better'

        res = bootstrap_paired(
            df, target_model, best, key, direction, metric_type,
            n_boot=n_boot, rng=rng,
        )
        results.append({
            'Dataset': dataset_name,
            'Metric': name,
            'Target': target_model,
            'Baseline': best,
            'Δ (target − baseline)': res['obs_diff'],
            '95% CI lower': res['ci_low'],
            '95% CI upper': res['ci_high'],
            'P value': res['p_value'],
            'N patients': res['n_patients'],
            'N cases': res['n_cases'],
            'Significant': res['p_value'] < 0.05,
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvs', nargs='+', required=True,
                        help='Path(s) to per-case CSVs, one per dataset. '
                             'Dataset name is inferred from the filename, '
                             'e.g. per_case_beijing_all.csv -> beijing.')
    parser.add_argument('--target', default='UPolarSeg',
                        help='Target model name (default: UPolarSeg)')
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='paired_test_results.csv')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    all_results = []
    for csv_path in args.csvs:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f'[WARN] Missing {csv_path}, skip.')
            continue

        # per_case_<name>_all.csv -> <name>
        stem = csv_path.stem
        if stem.startswith('per_case_'):
            stem = stem[len('per_case_'):]
        if stem.endswith('_all'):
            stem = stem[:-len('_all')]
        dataset_name = stem

        df = pd.read_csv(csv_path)
        print(f'\n===== {dataset_name} =====')
        print(f'Models: {sorted(df["model"].unique().tolist())}')
        print(f'Cases: {df["case_id"].nunique()}, '
              f'Patients: {df["patient_id"].nunique()}')

        res = run_dataset(df, dataset_name, args.target,
                          DEFAULT_TASKS, args.n_boot, rng)
        if not res.empty:
            all_results.append(res)
            print(res.to_string(index=False))

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv(args.output, index=False)
        print(f'\nSaved to {args.output}')
    else:
        print('\nNo results generated.')


if __name__ == '__main__':
    main()