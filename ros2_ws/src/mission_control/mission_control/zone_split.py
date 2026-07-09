"""Naive cellular decomposition baseline: equal-width vertical strips.

Splits the free-space polygon (boundary minus dead_zones) into N vertical
strips of equal width and hands each strip to the drone whose home position
sorts into the same left-to-right order. Strips can come out as MultiPolygon
if a dead_zone happens to cut one in two -- that's fine, coverage_plan.py
handles multi-part zones directly.

This is intentionally inefficient (doesn't account for area balance, drone
travel distance, etc.) -- it exists as a swappable baseline per the project
plan; a smarter decomposition algorithm will replace this module later.
"""
from shapely.geometry import Polygon, box


def build_free_space(boundary_points, dead_zone_point_lists):
    """boundary_points: [(x, y), ...] CCW. dead_zone_point_lists: [[(x, y), ...], ...]."""
    return Polygon(boundary_points, holes=dead_zone_point_lists)


def split_into_strips(free_space, num_zones):
    """Return a list of `num_zones` shapely geometries, ordered left (min x) to right."""
    minx, miny, maxx, maxy = free_space.bounds
    width = (maxx - minx) / num_zones
    strips = []
    for i in range(num_zones):
        strip_box = box(minx + i * width, miny - 1.0, minx + (i + 1) * width, maxy + 1.0)
        strips.append(free_space.intersection(strip_box))
    return strips


def assign_zones_to_drones(free_space, drones):
    """drones: list of dicts with keys 'id' and 'home_position' ([x, y, z]).

    Returns dict drone_id -> shapely geometry, matching strips (left to right)
    to drones (left to right by home x-coordinate).
    """
    strips = split_into_strips(free_space, len(drones))
    drones_by_x = sorted(drones, key=lambda d: d['home_position'][0])
    return {d['id']: strip for d, strip in zip(drones_by_x, strips)}
