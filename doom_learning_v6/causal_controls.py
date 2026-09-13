"""Pure mathematical controls for the causal imitation experiment."""

import math

import numpy as np

DEAD_ZONE = .1


def _direction(values):
    return np.where(values < -DEAD_ZONE, -1, np.where(values > DEAD_ZONE, 1, 0))


def _score_turns(predictions, targets):
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if p.ndim != 1 or t.ndim != 1 or not len(p) or p.shape != t.shape:
        raise ValueError('Prediction and target lengths must match')
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Finite nonempty one-dimensional turn values required')
    predicted, actual = _direction(p), _direction(t)
    support, recall = {}, {}
    for name, direction in [('left', -1), ('idle', 0), ('right', 1)]:
        selected = actual == direction
        support[name] = int(selected.sum())
        if support[name]:
            recall[name] = float(np.mean(predicted[selected] == direction))
    magnitude = max(float(np.mean(np.abs(t))), np.nextafter(DEAD_ZONE, math.inf))
    score = {'mae_degrees': float(np.mean(np.abs(p - t))),
             'direction_support': support, 'direction_recall': recall,
             'balanced_direction_recall': float(np.mean(list(recall.values())))}
    score['dead_zone_degrees'] = DEAD_ZONE
    score['constant_baselines'] = {
        name: {'turn_degrees': value, **_score_core(np.full_like(t, value), t)}
        for name, value in [('left', -magnitude), ('idle', 0.), ('right', magnitude)]}
    return score


def _score_core(predictions, targets):
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    predicted, actual = _direction(p), _direction(t)
    support, recall = {}, {}
    for name, direction in [('left', -1), ('idle', 0), ('right', 1)]:
        selected = actual == direction
        support[name] = int(selected.sum())
        if support[name]:
            recall[name] = float(np.mean(predicted[selected] == direction))
    return {'mae_degrees': float(np.mean(np.abs(p - t))),
            'direction_support': support, 'direction_recall': recall,
            'balanced_direction_recall': float(np.mean(list(recall.values())))}


def frame_ticks(count):
    if (isinstance(count, (bool, np.bool_)) or
            not isinstance(count, (int, np.integer)) or count <= 0 or
            int(count) > np.iinfo(np.int64).max):
        raise ValueError('Positive integer frame count required')
    count = int(count)
    boundaries = np.array([round(i * 10000 / 35) for i in range(count + 1)],
                          dtype=np.int64)
    return np.diff(boundaries)


def _schedule(currents, ticks):
    try:
        c = np.asarray(currents, dtype=np.float64)
        t = np.asarray(ticks)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Matching current/tick vectors required') from error
    if c.ndim != 1 or not len(c) or t.shape != c.shape or t.dtype.kind not in 'iu':
        raise ValueError('Matching current/tick vectors required')
    if t.dtype.kind == 'u' and np.any(t > np.iinfo(np.int64).max):
        raise ValueError('Tick values must fit signed 64-bit integers')
    if (not np.isfinite(c).all() or np.any(c < 0) or np.any(c > 4) or
            c[0] != 0 or np.any(t <= 0)):
        raise ValueError('Finite bounded current, first zero, positive ticks required')
    return c.copy(), t.astype(np.int64, copy=True)


def dose_proof(aligned, shifted, ticks):
    a, t = _schedule(aligned, ticks)
    s, _ = _schedule(shifted, ticks)
    groups = {}
    for duration in np.unique(t):
        selected = t == duration
        for dtype in (np.float64, np.float32):
            unsigned = np.uint64 if dtype == np.float64 else np.uint32
            x = np.sort(a[selected].astype(dtype).view(unsigned))
            y = np.sort(s[selected].astype(dtype).view(unsigned))
            if not np.array_equal(x, y):
                raise ValueError('Duration-conditioned amplitude multiset differs')
        groups[str(int(duration))] = int(selected.sum())
    totals = {}
    for name, dtype in [('requested', np.float64), ('float32', np.float32)]:
        original = math.fsum(float(v) * int(dt) * .1 for v, dt in zip(a.astype(dtype), t))
        replayed = math.fsum(float(v) * int(dt) * .1 for v, dt in zip(s.astype(dtype), t))
        if original != replayed:
            raise ValueError('Assigned current integral differs')
        totals[name + '_current_ms'] = original
    return {'exact': True, 'duration_groups': groups, **totals}


def duration_permutation(currents, ticks):
    c, t = _schedule(currents, ticks)
    mapping = np.arange(len(c), dtype=np.int64)
    for duration in np.unique(t[1:]):
        ix = np.flatnonzero((t == duration) & (mapping > 0))
        mapping[ix] = np.roll(ix, len(ix) // 2)
    shifted = c[mapping]
    if np.array_equal(mapping, np.arange(len(c))):
        raise ValueError('No nontrivial temporal permutation')
    dose_proof(c, shifted, t)
    return shifted, mapping


def diagnostic_score(predictions, targets):
    result = _score_turns(predictions, targets)
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    actual, predicted = _direction(t), _direction(p)
    classes = [('left', -1), ('idle', 0), ('right', 1)]
    per = {name: float(np.abs(p[actual == c] - t[actual == c]).mean())
           for name, c in classes if np.any(actual == c)}
    result.update(class_mae_degrees=per,
                  balanced_mae_degrees=float(np.mean(list(per.values())))
                  if per else float('nan'),
                  predicted_class_counts={name: int((predicted == c).sum())
                                          for name, c in classes},
                  confusion_rows_target_columns_prediction_left_idle_right=[
                      [int(((actual == a) & (predicted == b)).sum()) for _, b in classes]
                      for _, a in classes])
    blocks = []
    for ix in np.array_split(np.arange(len(t)), 4):
        if len(ix):
            blocks.append({'start': int(ix[0]), 'end_exclusive': int(ix[-1] + 1),
                           **_score_turns(p[ix], t[ix])})
    result['contiguous_blocks'] = blocks
    return result


def learned_direction(baseline, learned):
    b = np.asarray(baseline, dtype=np.float64)
    l = np.asarray(learned, dtype=np.float64)
    if (b.ndim != 1 or not len(b) or l.shape != b.shape or
            not np.isfinite(b).all() or not np.isfinite(l).all() or np.any(b <= 0)):
        raise ValueError('Finite positive baseline and matching learned vectors required')
    displacement = l / b - 1
    norm = float(np.max(np.abs(displacement)))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Nonzero finite displacement required')
    return displacement / norm


def initialize_efficacies(brain, memory):
    m = np.asarray(memory, dtype=np.float64)
    edges = np.asarray(brain.circuit['edges'])
    baseline = np.asarray(brain.baseline_plastic, dtype=np.float64)
    if getattr(brain, '_metal_batch_owner', None) is not None:
        raise ValueError('Manual initialization requires an unowned brain')
    if (m.shape != brain.memory_w.shape or m.shape != brain.memory_u.shape or
            not np.isfinite(m).all() or np.any(m < -.9) or np.any(m > 1)):
        raise ValueError('Matching bounded memory required')
    if (edges.ndim != 1 or baseline.shape != m.shape or
            np.any(edges < 0) or np.any(edges >= len(brain.weight)) or
            not np.isfinite(baseline).all() or np.any(baseline <= 0)):
        raise ValueError('Valid baseline plastic state required')
    weights = (baseline * (1 + m)).astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Finite positive efficacy required')
    brain.memory_u[:] = m
    brain.memory_w[:] = m
    brain.weight[edges] = weights
