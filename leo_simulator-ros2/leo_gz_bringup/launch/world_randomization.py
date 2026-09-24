"""Deterministic, collision-free task-object placement for mission worlds."""

import math
import random


TARGET_NAMES = ("red_box", "green_box", "blue_box")
DELIVERY_PREFIXES = (
    "red_delivery_box",
    "green_delivery_box",
    "blue_delivery_box",
)


def _seed(value):
    text = str(value).strip().lower()
    if text in {"", "auto", "random"}:
        return random.SystemRandom().randint(1, 2_147_483_647)
    return int(text)


def _xy(model):
    values = (model.findtext("pose") or "0 0 0 0 0 0").split()
    return float(values[0]), float(values[1])


def _set_xy(model, x, y):
    pose = model.find("pose")
    values = (pose.text or "0 0 0 0 0 0").split()
    while len(values) < 6:
        values.append("0")
    values[0], values[1] = f"{x:.6f}", f"{y:.6f}"
    pose.text = " ".join(values)


def _bounds(world):
    ground = next(
        model for model in world.findall("model")
        if model.get("name") == "ground_plane"
    )
    size = [
        float(value)
        for value in ground.findtext(".//collision/geometry/box/size").split()
    ]
    center_x, center_y = _xy(ground)
    margin = 0.65
    return (
        center_x - size[0] / 2.0 + margin,
        center_x + size[0] / 2.0 - margin,
        center_y - size[1] / 2.0 + margin,
        center_y + size[1] / 2.0 - margin,
    )


def _sample(rng, bounds, occupied, clearance, accept=None):
    min_x, max_x, min_y, max_y = bounds
    for _ in range(10_000):
        candidate = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
        if accept is not None and not accept(candidate):
            continue
        if all(
            math.hypot(candidate[0] - x, candidate[1] - y) >= clearance
            for x, y in occupied
        ):
            return candidate
    raise RuntimeError("Could not generate a collision-free task-object layout")


def _visually_separated(candidate, zone, center=(0.0, 0.0)):
    """Keep a pickup and same-color zone apart in angle and distance.

    This prevents the tall zone from occluding the compact pickup along the
    central sightlines from which robots commonly discover task objects.
    """
    candidate_vector = (candidate[0] - center[0], candidate[1] - center[1])
    zone_vector = (zone[0] - center[0], zone[1] - center[1])
    candidate_length = math.hypot(*candidate_vector)
    zone_length = math.hypot(*zone_vector)
    if candidate_length < 0.5 or zone_length < 0.5:
        return False
    cosine = (
        candidate_vector[0] * zone_vector[0]
        + candidate_vector[1] * zone_vector[1]
    ) / (candidate_length * zone_length)
    angular_separation = math.acos(max(-1.0, min(1.0, cosine)))
    pair_distance = math.hypot(
        candidate[0] - zone[0], candidate[1] - zone[1]
    )
    return angular_separation >= math.radians(75.0) and pair_distance >= 2.0


def randomize_task_models(world, mission, seed_value):
    """Randomize target zones and delivery-box clusters in an SDF world."""
    used_seed = _seed(seed_value)
    rng = random.Random(used_seed ^ 0x71A5C0DE)
    models = {model.get("name", ""): model for model in world.findall("model")}
    bounds = _bounds(world)
    occupied = []

    for name in TARGET_NAMES:
        model = models.get(name)
        if model is None:
            continue
        x, y = _sample(
            rng,
            bounds,
            occupied,
            clearance=1.5,
            accept=lambda candidate: math.hypot(*candidate) >= 0.75,
        )
        _set_xy(model, x, y)
        occupied.append((x, y))

    if mission == "delivery":
        for color, prefix in zip(("red", "green", "blue"), DELIVERY_PREFIXES):
            group = [
                model for name, model in models.items()
                if name == prefix or name.startswith(prefix + "_")
            ]
            if not group:
                continue
            base = models[prefix]
            base_x, base_y = _xy(base)
            offsets = [(model, _xy(model)[0] - base_x, _xy(model)[1] - base_y) for model in group]
            zone = _xy(models[f"{color}_box"])
            x, y = _sample(
                rng,
                bounds,
                occupied,
                clearance=1.5,
                accept=lambda candidate, zone=zone: _visually_separated(
                    candidate, zone
                ),
            )
            for model, dx, dy in offsets:
                _set_xy(model, x + dx, y + dy)
            occupied.append((x, y))

    return used_seed
