"""Statistics and CSV-oriented aggregation for OD/PPA evaluation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CONTINUOUS_SPECS = [
    ("OD", "all", "dsc", "od_dsc", True),
    ("OD", "all", "nsd", "od_nsd", True),
    ("OD", "pred_nonempty", "hd95_px", "od_hd95_px", False),
    ("OD", "pred_nonempty", "hd95_norm", "od_hd95_norm", False),
    ("PPA", "positive", "dsc", "ppa_dsc", True),
    ("PPA", "positive", "nsd", "ppa_nsd", True),
    ("PPA", "positive_pred_nonempty", "hd95_px", "ppa_hd95_px", False),
    ("PPA", "positive_pred_nonempty", "hd95_norm", "ppa_hd95_norm", False),
    ("PPA", "negative", "fp_norm", "ppa_fp_norm", False),
]

BINARY_SPECS = [
    ("PPA", "negative", "specificity", "ppa_negative_correct", True),
    ("PPA", "positive", "empty_prediction_rate", "ppa_pred_empty", False),
    ("OD", "all", "empty_prediction_rate", "od_pred_empty", False),
]


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _eligible(rec: Mapping, target: str, subgroup: str, key: str) -> bool:
    if target == "OD":
        if not bool(rec.get("od_reference_valid", False)):
            return False
        if subgroup == "pred_nonempty" and not bool(rec.get("od_hd95_valid", False)):
            return False
        return True
    status = rec.get("ppa_status")
    if subgroup == "positive":
        return status == "positive"
    if subgroup == "positive_pred_nonempty":
        return status == "positive" and bool(rec.get("ppa_hd95_valid", False))
    if subgroup == "negative":
        return status == "negative"
    return False


def _groups(records: Sequence[Mapping], idx: Sequence[int], cluster_key: str) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i in idx:
        cid = str(records[i].get(cluster_key) or records[i].get("case_id") or i)
        groups.setdefault(cid, []).append(i)
    return groups


def _cluster_bootstrap_mean(
    records: Sequence[Mapping], idx: Sequence[int], key: str, cluster_key: str,
    n_boot: int, seed: int,
) -> Tuple[float, float]:
    groups = _groups(records, idx, cluster_key)
    keys = list(groups)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.asarray([float(records[i][key]) for i in idx], dtype=np.float64)
    # Fast path: precompute per-cluster sums/counts, preserving image-weighted mean.
    sums, counts = [], []
    for g in keys:
        v = [float(records[i][key]) for i in groups[g]]
        sums.append(float(np.sum(v)))
        counts.append(len(v))
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    stats = np.empty(int(n_boot), dtype=np.float64)
    ng = len(keys)
    for b in range(int(n_boot)):
        pick = rng.integers(0, ng, size=ng)
        stats[b] = sums[pick].sum() / counts[pick].sum()
    return tuple(np.percentile(stats, [2.5, 97.5]).tolist())


def _cluster_bootstrap_proportion(
    records: Sequence[Mapping], idx: Sequence[int], key: str, cluster_key: str,
    n_boot: int, seed: int,
) -> Tuple[float, float]:
    return _cluster_bootstrap_mean(records, idx, key, cluster_key, n_boot, seed)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def summarize_records(records: Sequence[Mapping], *, n_boot=10000, seed=42, cluster_key="patient_id") -> List[dict]:
    rows: List[dict] = []
    for j, (target, subgroup, metric, key, higher) in enumerate(CONTINUOUS_SPECS):
        eligible = [i for i, r in enumerate(records) if _eligible(r, target, subgroup, key) and _finite(r.get(key))]
        vals = np.asarray([float(records[i][key]) for i in eligible], dtype=np.float64)
        n_patients = len(_groups(records, eligible, cluster_key)) if eligible else 0
        if vals.size:
            estimate = float(vals.mean())
            sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
            if n_patients >= 2:
                lo, hi = _cluster_bootstrap_mean(records, eligible, key, cluster_key, n_boot, seed + j)
                ci_method, ci_status = "cluster_percentile_bootstrap", "ok"
            else:
                lo = hi = float("nan")
                ci_method, ci_status = "cluster_percentile_bootstrap", "insufficient_clusters"
        else:
            estimate = sd = lo = hi = float("nan")
            ci_method, ci_status = "cluster_percentile_bootstrap", "no_eligible_cases"
        rows.append({
            "target": target, "subgroup": subgroup, "metric": metric,
            "unit": "px" if metric == "hd95_px" else "ratio" if metric in {"hd95_norm", "fp_norm"} else "",
            "higher_is_better": bool(higher), "estimate": estimate, "sd": sd,
            "ci_lower": lo, "ci_upper": hi, "ci_method": ci_method, "ci_status": ci_status,
            "n_eligible_cases": int(vals.size), "n_valid_cases": int(vals.size), "n_patients": n_patients,
            "numerator": "", "denominator": "",
        })

    base = len(rows)
    for j, (target, subgroup, metric, key, higher) in enumerate(BINARY_SPECS):
        eligible = [i for i, r in enumerate(records) if _eligible(r, target, subgroup, key) and _finite(r.get(key))]
        vals = np.asarray([float(records[i][key]) for i in eligible], dtype=np.float64)
        n_patients = len(_groups(records, eligible, cluster_key)) if eligible else 0
        if vals.size:
            estimate = float(vals.mean())
            successes = int(np.rint(vals.sum()))
            independent = n_patients == len(eligible)
            if independent:
                lo, hi = wilson_interval(successes, len(eligible))
                ci_method, ci_status = "wilson", "ok"
            elif n_patients >= 2 and not (np.all(vals == 0) or np.all(vals == 1)):
                lo, hi = _cluster_bootstrap_proportion(
                    records, eligible, key, cluster_key, n_boot, seed + base + j
                )
                ci_method, ci_status = "cluster_percentile_bootstrap", "ok"
            else:
                lo = hi = float("nan")
                ci_method = "cluster_percentile_bootstrap"
                ci_status = "degenerate_or_insufficient_clusters"
        else:
            estimate = lo = hi = float("nan")
            successes = 0
            ci_method, ci_status = "wilson", "no_eligible_cases"
        rows.append({
            "target": target, "subgroup": subgroup, "metric": metric, "unit": "",
            "higher_is_better": bool(higher), "estimate": estimate, "sd": float("nan"),
            "ci_lower": lo, "ci_upper": hi, "ci_method": ci_method, "ci_status": ci_status,
            "n_eligible_cases": len(eligible), "n_valid_cases": len(eligible), "n_patients": n_patients,
            "numerator": successes, "denominator": len(eligible),
        })
    return rows


def _paired_cluster_bootstrap_delta(
    current: Sequence[Mapping], reference: Sequence[Mapping], pairs: Sequence[Tuple[int, int]],
    key: str, cluster_key: str, n_boot: int, seed: int,
) -> Tuple[float, float]:
    # Cluster membership is taken from current after strict case matching.
    by_cluster: Dict[str, List[Tuple[int, int]]] = {}
    for i, j in pairs:
        c = str(current[i].get(cluster_key) or current[i].get("case_id"))
        by_cluster.setdefault(c, []).append((i, j))
    clusters = list(by_cluster)
    if len(clusters) < 2:
        return float("nan"), float("nan")
    sums, counts = [], []
    for c in clusters:
        ds = [float(current[i][key]) - float(reference[j][key]) for i, j in by_cluster[c]]
        sums.append(float(np.sum(ds))); counts.append(len(ds))
    sums = np.asarray(sums); counts = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        stats[b] = sums[pick].sum() / counts[pick].sum()
    return tuple(np.percentile(stats, [2.5, 97.5]).tolist())


def _paired_cluster_signflip_p(
    current: Sequence[Mapping], reference: Sequence[Mapping], pairs: Sequence[Tuple[int, int]],
    key: str, cluster_key: str, n_perm: int, seed: int,
) -> float:
    by_cluster: Dict[str, List[float]] = {}
    for i, j in pairs:
        c = str(current[i].get(cluster_key) or current[i].get("case_id"))
        by_cluster.setdefault(c, []).append(float(current[i][key]) - float(reference[j][key]))
    if not by_cluster:
        return float("nan")
    clusters = list(by_cluster)
    sums = np.asarray([np.sum(by_cluster[c]) for c in clusters], dtype=np.float64)
    counts = np.asarray([len(by_cluster[c]) for c in clusters], dtype=np.float64)
    observed = float(sums.sum() / counts.sum())
    if np.allclose(sums, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    total_count = counts.sum()
    for _ in range(int(n_perm)):
        signs = rng.choice((-1.0, 1.0), size=len(clusters))
        stat = float((signs * sums).sum() / total_count)
        if abs(stat) >= abs(observed) - 1e-15:
            extreme += 1
    return float((extreme + 1) / (int(n_perm) + 1))


def exact_mcnemar_p(current_vals: Sequence[int], reference_vals: Sequence[int]) -> float:
    b = sum(1 for c, r in zip(current_vals, reference_vals) if c == 1 and r == 0)
    c = sum(1 for c, r in zip(current_vals, reference_vals) if c == 0 and r == 1)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def compare_records(
    current: Sequence[Mapping], reference: Sequence[Mapping], *, n_boot=10000,
    n_perm=100000, seed=42, cluster_key="patient_id",
) -> List[dict]:
    ref_by_case = {str(r["case_id"]): (j, r) for j, r in enumerate(reference)}
    cur_cases = [str(r["case_id"]) for r in current]
    if len(set(cur_cases)) != len(cur_cases):
        raise ValueError("Current per_case contains duplicate case_id values")
    if set(cur_cases) != set(ref_by_case):
        missing = sorted(set(cur_cases) - set(ref_by_case))[:10]
        extra = sorted(set(ref_by_case) - set(cur_cases))[:10]
        raise ValueError(f"Reference case set differs. missing_in_ref={missing}, extra_in_ref={extra}")
    for r in current:
        rr = ref_by_case[str(r["case_id"])][1]
        cid = r["case_id"]
        if str(r.get("ppa_status")) != str(rr.get("ppa_status")):
            raise ValueError(f"PPA status mismatch for case {cid}")
        if str(r.get("protocol_version")) != str(rr.get("protocol_version")):
            raise ValueError(f"Protocol version mismatch for case {cid}")
        for key in ("gt_od_area_px", "gt_ppa_area_px", "eval_height", "eval_width"):
            if _finite(r.get(key)) and _finite(rr.get(key)) and float(r[key]) != float(rr[key]):
                raise ValueError(f"Reference GT/evaluation geometry mismatch for case {cid}: {key}")
        if _finite(r.get("nsd_tolerance_px")) and _finite(rr.get("nsd_tolerance_px")):
            if not np.isclose(float(r["nsd_tolerance_px"]), float(rr["nsd_tolerance_px"]), rtol=0, atol=1e-8):
                raise ValueError(f"NSD tolerance mismatch for case {cid}")
        cur_cluster = str(r.get(cluster_key) or r.get("case_id"))
        ref_cluster = str(rr.get(cluster_key) or rr.get("case_id"))
        if cur_cluster != ref_cluster:
            raise ValueError(f"Cluster ID mismatch for case {cid}: {cur_cluster!r} vs {ref_cluster!r}")

    rows = []
    specs = CONTINUOUS_SPECS
    for sidx, (target, subgroup, metric, key, higher) in enumerate(specs):
        pairs = []
        for i, r in enumerate(current):
            j, rr = ref_by_case[str(r["case_id"])]
            if (_eligible(r, target, subgroup, key) and _eligible(rr, target, subgroup, key)
                    and _finite(r.get(key)) and _finite(rr.get(key))):
                pairs.append((i, j))
        if pairs:
            curv = np.asarray([float(current[i][key]) for i, _ in pairs])
            refv = np.asarray([float(reference[j][key]) for _, j in pairs])
            delta = float((curv - refv).mean())
            lo, hi = _paired_cluster_bootstrap_delta(
                current, reference, pairs, key, cluster_key, n_boot, seed + sidx
            )
            p = _paired_cluster_signflip_p(
                current, reference, pairs, key, cluster_key, n_perm, seed + 1000 + sidx
            )
        else:
            curv = refv = np.asarray([])
            delta = lo = hi = p = float("nan")
        rows.append({
            "target": target, "subgroup": subgroup, "metric": metric,
            "higher_is_better": bool(higher), "n_paired_cases": len(pairs),
            "n_paired_patients": len(_groups(current, [i for i, _ in pairs], cluster_key)) if pairs else 0,
            "estimate_current": float(curv.mean()) if curv.size else float("nan"),
            "estimate_reference": float(refv.mean()) if refv.size else float("nan"),
            "delta_current_minus_reference": delta,
            "delta_ci_lower": lo, "delta_ci_upper": hi,
            "test_method": "paired_cluster_signflip_permutation",
            "p_raw": p, "p_adjusted": float("nan"), "adjustment_status": "not_applied_single_run",
        })

    # Specificity comparison: exact McNemar only when each patient contributes one eligible case.
    pairs = []
    for i, r in enumerate(current):
        j, rr = ref_by_case[str(r["case_id"])]
        if (_eligible(r, "PPA", "negative", "ppa_negative_correct")
                and _eligible(rr, "PPA", "negative", "ppa_negative_correct")
                and _finite(r.get("ppa_negative_correct")) and _finite(rr.get("ppa_negative_correct"))):
            pairs.append((i, j))
    curvals = [int(float(current[i]["ppa_negative_correct"])) for i, _ in pairs]
    refvals = [int(float(reference[j]["ppa_negative_correct"])) for _, j in pairs]
    n_patients = len(_groups(current, [i for i, _ in pairs], cluster_key)) if pairs else 0
    if pairs and n_patients == len(pairs):
        p = exact_mcnemar_p(curvals, refvals)
        method = "exact_mcnemar"
    elif pairs:
        p = _paired_cluster_signflip_p(
            current, reference, pairs, "ppa_negative_correct", cluster_key, n_perm, seed + 9000
        )
        method = "paired_cluster_signflip_permutation"
    else:
        p = float("nan"); method = "not_available"
    if pairs:
        delta = float(np.mean(curvals) - np.mean(refvals))
        lo, hi = _paired_cluster_bootstrap_delta(
            current, reference, pairs, "ppa_negative_correct", cluster_key, n_boot, seed + 8000
        )
    else:
        delta = lo = hi = float("nan")
    rows.append({
        "target": "PPA", "subgroup": "negative", "metric": "specificity",
        "higher_is_better": True, "n_paired_cases": len(pairs), "n_paired_patients": n_patients,
        "estimate_current": float(np.mean(curvals)) if pairs else float("nan"),
        "estimate_reference": float(np.mean(refvals)) if pairs else float("nan"),
        "delta_current_minus_reference": delta, "delta_ci_lower": lo, "delta_ci_upper": hi,
        "test_method": method, "p_raw": p, "p_adjusted": float("nan"),
        "adjustment_status": "not_applied_single_run",
    })
    return rows
