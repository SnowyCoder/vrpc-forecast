
from typing import List, Tuple, NamedTuple
import numpy as np


MAX_CAPACITY_PER_VEHICLE = 50_000  # In Kilograms

# If a customer should receive 137 Kg and we use a granularity of 100 we round it up to 200Kg
# This is only useful for the column generator method because the subproblem complexity depends
# quadratically on the maximum capacity, so by increasing the "grain size" we can improve performance by a lot
GRANULARITY = 1000  # In Kilograms per customer
MAX_VEHICLES = 20


class FacilityData(NamedTuple):
    name: str
    location: np.array
    material: float


class ProblemData(NamedTuple):
    facilities: List[FacilityData]
    distribution_center: np.array
    max_vehicle_capacity: float


def points_to_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    c = np.array([[complex(p[0], p[1]) for p in points]])
    return abs(c.T - c)
