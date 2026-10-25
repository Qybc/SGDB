"""Zero-training mask-axis operand; all object identities and locations stay frozen."""
import copy
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

CANDIDATES = [
    {'name': f'{assignment}_trim{trim}', 'assignment': assignment, 'trim': trim}
    for assignment in ['components', 'box_voronoi'] for trim in [0., .01]
]


def replace_axes(objects, mask, original_size, candidate, target_class=2):
    result = copy.deepcopy(objects)
    indices = [i for i, o in enumerate(objects) if o['class'] == target_class]
    changes = []
    if not indices:
        return result, changes
    height, width = mask.shape
    image_size = np.asarray(original_size, dtype=float)
    centers = np.array([objects[i]['center'] for i in indices])
    yy, xx = np.nonzero(mask)
    points = np.stack([(xx + .5) / width, (yy + .5) / height], axis=1)
    chosen = [None] * len(indices)
    if candidate['assignment'] == 'components':
        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        components = []
        for label in range(1, count + 1):
            cy, cx = np.nonzero(labels == label)
            if len(cx) >= 16:
                p = np.stack([(cx + .5) / width, (cy + .5) / height], axis=1)
                components.append(p)
        if len(components) != len(indices):
            return result, changes
        midpoints = np.array([p.mean(0) for p in components])
        distances = np.linalg.norm(centers[:, None] - midpoints[None], axis=2)
        aa, bb = linear_sum_assignment(distances)
        for a, b in zip(aa, bb):
            if distances[a, b] <= .04:
                chosen[a] = components[b]
    elif candidate['assignment'] == 'box_voronoi':
        costs = np.full((len(indices), len(points)), np.inf)
        for a, i in enumerate(indices):
            box = np.array(objects[i]['box']); middle = (box[:2] + box[2:]) / 2
            size = np.maximum(box[2:] - box[:2], .02)
            inside = np.all(np.abs(points - middle) <= size * .625, axis=1)
            costs[a, inside] = np.sum(((points[inside] - centers[a]) / size) ** 2, axis=1)
        if len(points):
            owner = costs.argmin(0)
            valid = np.isfinite(costs.min(0))
            for a in range(len(indices)):
                chosen[a] = points[valid & (owner == a)]
    else:
        raise ValueError(candidate['assignment'])
    for a, i in enumerate(indices):
        p = chosen[a]
        if p is None or len(p) < 16:
            continue
        box = np.array(objects[i]['box']); size = np.maximum(box[2:] - box[:2], .02)
        within = np.all((p >= box[:2] - size * .125) & (p <= box[2:] + size * .125), axis=1)
        if within.mean() < .75 or np.linalg.norm(p.mean(0) - centers[a]) > .04:
            continue
        native = p * image_size
        middle = native.mean(0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov((native - middle).T))
        if eigenvalues[-1] / max(eigenvalues[0], 1e-12) < 4:
            continue
        vector = eigenvectors[:, -1]
        projected = (native - middle) @ vector
        lo, hi = np.quantile(projected, [candidate['trim'], 1 - candidate['trim']])
        axis = np.stack([middle + lo * vector, middle + hi * vector]) / image_size
        if not np.isfinite(axis).all():
            continue
        result[i]['axis'] = axis.tolist()
        changes.append(i)
    for old, new in zip(objects, result):
        assert {k: v for k, v in old.items() if k != 'axis'} == {k: v for k, v in new.items() if k != 'axis'}
        if old['class'] != target_class:
            assert old == new
    return result, changes
