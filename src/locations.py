from hashlib import sha256
from typing import Dict, NamedTuple
import pandas as pd
from numpy import random, linalg
import numpy as np

from common import DATA_DIR

# The map goes from 0 to
MAP_SIZE = 1000
MIN_REGION_DIST = 200
MIN_CITY_DIST = 30
MIN_CENTER_DIST = 10
CENTER_LOCATIONS_DEFAULT_SEED = 'generate_center_locations.001'


class Locations(NamedTuple):
    df: pd.DataFrame
    regions: Dict[int, np.array]
    cities: Dict[int, np.array]
    centers: Dict[int, np.array]
    distribution_center: np.array


def generate_center_locations(seed: str=None) -> Locations:
    df = pd.read_csv(DATA_DIR / 'fulfilment_center_info.csv')
    # There are:
    # - 77 centers
    # - 51 cities
    # - 8 regions
    # Each center is located in a city (which is in a region)
    # The idea is to make centers that share the same region near each other
    # and those who share the same city even nearer.
    # To generate it we're using hierarchical placing with some checks that guarantee that
    # we don't place points too near each other, using Gaussians all over the place
    seed = sha256((seed or CENTER_LOCATIONS_DEFAULT_SEED).encode()).digest()
    rng = random.default_rng(int.from_bytes(seed, 'little'))

    def checkdist(point, others, min_dist):
        """ Check that a point is in the map bounds and that it's at least distant min_dist from all nodes in the others set"""
        if not all(0 <= x < MAP_SIZE for x in point):
            return False
        return min([linalg.norm(x - point) for x in others], default=MAP_SIZE) >= min_dist

    regions = {}
    print("[center_locs] Generating regions")
    for region in df['region_code'].unique():
        while True:
            p = rng.normal(MAP_SIZE / 2, MAP_SIZE / 2, 2)

            if checkdist(p, regions.values(), MIN_REGION_DIST):
                break

        regions[region] = p

    cities = {}
    print("[center_locs] Generating cities")
    # If a region is big their cities will be sparser
    # If it's small (i.e. if it has only one or two cities) its cities will be more concentrated
    for city in df['city_code'].unique():
        reg = df[df['city_code'] == city]['region_code'].iloc[0]
        region_size = (df['region_code'] == reg).sum()
        region_pos = regions[reg]
        while True:
            # We can control the "sparsity" by changing the Gaussian scale, but if it's too narrow
            # we'll never pass the MIN_CITY_DIST check.
            # (these values are eyeballed based on plotting, don't read too much into them)
            p = region_pos + rng.normal(0, region_size * 2.5 + 10, 2)

            if checkdist(p, cities.values(), MIN_CITY_DIST):
                break

        cities[city] = p

    centers = {}
    print("[center_locs] Generating centers")
    for idx, center in df.iterrows():
        city_pos = cities[center['city_code']]
        while True:
            p = city_pos + rng.normal(0, MIN_REGION_DIST / 8, 2)

            if checkdist(p, centers.values(), MIN_CENTER_DIST):
                break

        centers[center['center_id']] = p

    # Where should the distribution center be?
    # I imagine that it's more likely to be in a larger city
    # To simulate this behaviour, select randomly a center and generate another location in the same city
    # This should give a better chance to biffer cities
    i = rng.integers(0, len(df), 1)[0]
    city_pos = cities[df.iloc[i]['city_code']]
    while True:
        p = city_pos + rng.normal(0, MIN_REGION_DIST / 8, 2)

        if checkdist(p, centers.values(), MIN_CENTER_DIST):
            break
    distribution_center = p

    print("[center_locs] Generated!")
    return Locations(df, regions, cities, centers, distribution_center)


def plot_locations(locs: Locations):
    from matplotlib import pyplot

    print("[plot_locs] Plotting")
    # Regions: blue & annotated
    pyplot.scatter(*zip(*locs.regions.values()), s=400, alpha=0.5)
    for k in locs.regions:
        pyplot.annotate(str(k), locs.regions[k], ha='center', va='center')

    # Cities: red
    pyplot.scatter(*zip(*locs.cities.values()), c='r', s=100)

    df = locs.df
    # Centers: green (with variable size)
    for _idx, row in df.iterrows():
        p = locs.centers[row['center_id']]
        pyplot.scatter(*p, row['op_area']*10, c='g')

    pyplot.scatter(*locs.distribution_center, 200, c='y', alpha=0.7)

    pyplot.show()
