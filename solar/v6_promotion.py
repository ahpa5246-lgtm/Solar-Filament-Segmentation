"""Fail-closed promotion decision for the fixed V6 NMS experiment.

This module only evaluates recorded evidence. It neither runs inference nor
uses Kaggle credentials. A passing result is eligibility, not an official score.
"""
import math


def _finite_number(value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _count(value, minimum=0):
    return type(value) is int and value >= minimum


def promotion_decision(report):
    """Return a conservative, inspectable promotion verdict.

    Aggregates alone cannot prove date isolation: the private Kaggle run
    must produce observation-level evidence separately.
    """
    reasons = []

    def check(condition, reason):
        if not condition:
            reasons.append(reason)

    report = report if isinstance(report, dict) else {}
    result = report.get('result', {})
    result = result if isinstance(result, dict) else {}
    base = result.get('confirmation_baseline', {})
    base = base if isinstance(base, dict) else {}
    candidate = result.get('confirmation_candidate', {})
    candidate = candidate if isinstance(candidate, dict) else {}
    interval = result.get('paired_macro_delta_ci', {})
    interval = interval if isinstance(interval, dict) else {}

    check(report.get('status') == 'COMPLETE' and result.get('status') == 'completed',
          'Kaggle confirmation execution is not complete')
    check(all(report.get(k) is False for k in ('training', 'test_inference', 'official_submission')),
          'Confirmation must be inference-only and not an official submission')
    check(result.get('fixed_nms_confirmation') is True and
          result.get('pilot_observations_per_subset') is None,
          'Expected the fixed-NMS full confirmation, not the small pilot')
    check(result.get('baseline') == '1024|union' and
          result.get('selected') == '1024|nms',
          'Experiment settings differ from preregistered comparison')

    tune = result.get('tuning_observations')
    confirm = result.get('confirmation_observations')
    excluded = result.get('pilot_confirmation_excluded')
    processed = result.get('processed_observations')
    check(_count(tune, 40) and _count(confirm, 40) and _count(excluded, 16),
          'Insufficient date-separated tuning, confirmation, or pilot exclusion')
    check(_count(processed) and _count(tune) and _count(confirm) and
          processed == tune + confirm,
          'Incomplete observation processing')
    check(_finite_number(result.get('detector_cache_parity_min'), 0.99, 1.0),
          'Saved detector output parity is missing or too low')
    check(_count(base.get('physical_observations')) and
          base.get('physical_observations') == confirm and
          candidate.get('physical_observations') == confirm,
          'Metric sample counts differ from confirmation inventory')

    old = base.get('reference_micro_pq')
    new = candidate.get('reference_micro_pq')
    check(_finite_number(old, 0, 1) and _finite_number(new, 0, 1) and new > old,
          'Reference micro PQ did not strictly improve')
    low = interval.get('low')
    high = interval.get('high')
    estimate = interval.get('estimate')
    check(_finite_number(low, -1, 1) and _finite_number(high, -1, 1) and
          _finite_number(estimate, -1, 1) and 0 < low <= estimate <= high,
          'Paired macro improvement does not have a strictly positive confidence interval')
    check(_count(interval.get('replicates'), 1000),
          'Paired confidence interval has insufficient bootstrap replicates')
    check(result.get('confirmation_gain_supported') is True,
          'Recorded experiment did not support promotion')

    return {
        'promote': not reasons,
        'reasons': reasons,
        'experiment': 'V6 fixed detector mask NMS versus V5 union',
        'scope': 'Local validation; not an official Kaggle score',
        'confirmation_observations': confirm,
        'micro_pq_baseline': old,
        'micro_pq_candidate': new,
        'paired_macro_delta_ci': {'low': low, 'estimate': estimate, 'high': high},
    }
