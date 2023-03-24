
from typing import List, Tuple, NamedTuple
import numpy as np


MAX_CAPACITY_PER_VEHICLE = 50_000  # In Kilograms

# If a customer should receive 137 Kg and we use a granularity of 100 we round it up to 200Kg
# This is only useful for the column generator method because the subproblem complexity depends
# quadratically on the maximum capacity, so by increasing the "grain size" we can improve performance by a lot
# To be fair we impose these restrictions in all the algorithms
GRANULARITY = 1000  # In Kilograms per customer
DISTANCE_GRANULARITY = 100  # How many precision digits should we compute
MAX_VEHICLES = 20


class FacilityData(NamedTuple):
    name: str
    location: np.array
    material: float


class ProblemData(NamedTuple):
    facilities: List[FacilityData]
    distribution_center: np.array


class ProblemSolution(NamedTuple):
    routes: List[List[int]]
    total_cost: float

    def encode(self, data: ProblemData) -> str:
        return '_'.join('-'.join(map(lambda i: data.facilities[i].name, r)) for r in self.routes)


def points_to_distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    c = np.array([[complex(p[0], p[1]) for p in points]])
    return abs(c.T - c)
