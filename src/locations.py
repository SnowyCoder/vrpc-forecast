from hashlib import sha256
from math import sqrt
from typing import Dict, NamedTuple, List
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
    # This should give a better chance to bigger cities
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
    regions = pyplot.scatter(*zip(*locs.regions.values()), s=400, c='#FFFD00')
    for k in locs.regions:
        pyplot.annotate(str(k), locs.regions[k], ha='center', va='center')

    # Cities: red
    cities = pyplot.scatter(*zip(*locs.cities.values()), c='#FFAA00', s=100)

    df = locs.df
    centers = None
    # Centers: green (with variable size)
    for _idx, row in df.iterrows():
        p = locs.centers[row['center_id']]
        centers = pyplot.scatter(*p, row['op_area']*10, c='#FF0700')

    depot = pyplot.scatter(*locs.distribution_center, 200, c='#1240AB', alpha=0.7)

    pyplot.legend(
        (regions, cities, centers, depot),
        ('Region', 'City', 'Center', 'Depot'),
        scatterpoints=1
    )

    pyplot.show()


def plot_solution(locs: Locations, routes: List[List[int]]):
    from matplotlib import pyplot

    depot = pyplot.scatter(*locs.distribution_center, 200, c='#1240AB', alpha=0.7)

    df = locs.df
    centers = None
    # Centers: green (with variable size)
    for _idx, row in df.iterrows():
        p = locs.centers[row['center_id']]
        centers = pyplot.scatter(*p, row['op_area'] * 10, c='#FF0700')

    tot_dist = 0
    for route in routes:
        route = route
        x, y = [], []
        x.append(locs.distribution_center[0])
        y.append(locs.distribution_center[1])
        for i in route:
            x.append(locs.centers[i][0])
            y.append(locs.centers[i][1])

        x.append(locs.distribution_center[0])
        y.append(locs.distribution_center[1])

        dist = 0
        for (px, nx), (py, ny) in zip(zip(x[:-1], x[1:]), zip(y[:-1], y[1:])):
            dist += sqrt((px - nx)**2 + (py - ny)**2)
        tot_dist += dist
        pyplot.plot(x, y)

    pyplot.legend(
        (centers, depot),
        ('Center', 'Depot'),
        scatterpoints=1
    )

    text = f'Cost: {tot_dist:.2f}'
    pyplot.figtext(0.5, 0.01, text, horizontalalignment='center', fontsize=15)
    pyplot.show()
