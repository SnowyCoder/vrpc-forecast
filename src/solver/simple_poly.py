from .common import ProblemData, points_to_distance_matrix, MAX_CAPACITY_PER_VEHICLE, MAX_VEHICLES, GRANULARITY, \
    DISTANCE_GRANULARITY, ProblemSolution
from gurobipy import Model, GRB, quicksum as qsum
import numpy as np
from math import ceil, floor


def solve_flow_poly(data: ProblemData, subproblem: int | None) -> ProblemSolution | None:
    # Index 0 is the starting and ending point
    facilities = data.facilities
    if subproblem is not None:
        facilities = facilities[:subproblem]

    d = points_to_distance_matrix(
        [data.distribution_center] + [p.location for p in facilities]
    )
    d = np.ceil(DISTANCE_GRANULARITY * d)

    w = [0] + [ceil(f.material / GRANULARITY) for f in facilities]
    max_cap = floor(MAX_CAPACITY_PER_VEHICLE / GRANULARITY)

    V = range(len(facilities) + 1)
    W = [(i, j) for i in V for j in V if i != j]

    m = Model("cvrp_flow_poly")
    # x[i,j]=1 If edge i->j is selected
    # This is the asymmetric view of the problem, suboptimal (the problem here is symmetric)
    x = {(i, j): m.addVar(name=f'x[{i},{j}]', vtype=GRB.BINARY) for i in V for j in V if i != j}
    # Capacity at node i
    c = [m.addVar(name=f'c[{i}]', vtype=GRB.CONTINUOUS) for i in V]

    m.setObjective(qsum(d[i, j]*x[i, j] for (i, j) in W), GRB.MINIMIZE)

    # ~~~ Flow bounds ~~~
    # Every node should have exactly one entry path and one exit path
    m.addConstrs(qsum(x[i, j] for i in V if j != i) == 1 for j in V if j != 0)
    m.addConstrs(qsum(x[i, j] for j in V if j != i) == 1 for i in V if i != 0)

    # We should have the same number of vehicles departing and returning
    m.addConstr(qsum(x[0, i] - x[i, 0] for i in V if i != 0) == 0)

    # Limit the number of available vehicles
    m.addConstr(qsum(x[0, i] for i in V if i != 0) <= MAX_VEHICLES)

    # Connectivity + Capacity constraint:
    # Every truck starts with capacity = 0 and every node should add its weight to the capacity
    # This also doubles as a connectivity constraint. (I hope this works, but even if it does, it will never scale)
    m.addConstr(c[0] == 0)
    # If x[i,j] = 1, then
    # | c[j] - c[i] >= weight of j
    # otherwise:
    # | c[j] - c[i] >= w[j] - M
    M = 2*sum(w)
    m.addConstrs(c[j] - c[i] >= w[j] - M*(1-x[i, j]) for i in V for j in V if i != j and j != 0)
    m.addConstrs(c[i] <= max_cap for i in V)

    try:
        m.optimize()
    except KeyboardInterrupt:
        print("Interrupted")

    next_nodes = lambda i: [j for j in V if i != j and x[i, j].x > 0.5]

    def get_path(i: int):
        path = [i - 1]
        while i := next_nodes(i)[0]:
            path.append(i - 1)
        return path

    if m.status == GRB.Status.OPTIMAL or m.status == GRB.Status.INTERRUPTED:
        cost = m.objVal / DISTANCE_GRANULARITY
        routes = []
        print(f"Problem solved, cost: {cost}")
        start_paths = next_nodes(0)
        for p in start_paths:
            routes.append(get_path(p))
        # sometimes gurobi does not terminate because it needs memory to free memory, better be safe than sorry
        print(routes, cost)
        return ProblemSolution(routes, cost)
    else:
        print("No feasible solution found")
        return None
