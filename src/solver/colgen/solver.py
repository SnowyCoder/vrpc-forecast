from .optimize import derive_clients_count, optimize_problem
from ..common import ProblemData, points_to_distance_matrix, MAX_CAPACITY_PER_VEHICLE, GRANULARITY, DISTANCE_GRANULARITY
from .data import Context, CGProblemData, Route
from .branchnprice import branchnprice, debug_explore

from gurobipy import Model, quicksum as qsum, GRB
from typing import List
from math import floor, ceil
import numpy as np


def initial_sol(ctx: Context) -> List[Route]:
    dist = ctx.data.dists
    cap = ctx.data.cap
    max_cap = ctx.data.max_cap
    nlen = dist.shape[0]
    visited = [False] * nlen

    def closest_unexpl(p: int) -> int | None:
        mdist = 100**100
        mp = None
        for i in range(nlen):
            if not visited[i] and dist[i, p] < mdist:
                mdist = dist[i, p]
                mp = i
        return mp

    visited[0] = True
    routes = []

    path = [0]
    parent = 0
    curr_cap = 0
    curr_cost = 0

    def close_path():
        nonlocal curr_cap, curr_cost, path, node, parent
        if curr_cap == 0:
            return
        path.append(0)
        routes.append(ctx.add_route(path, False, curr_cost + dist[parent, 0]))
        path = [0]
        parent, node = 0, 0
        curr_cap, curr_cost = 0, 0

    while True:
        node = closest_unexpl(parent)
        if node is None:
            break
        if curr_cap + cap[node] > max_cap:
            if curr_cap == 0:
                raise RuntimeError('Impossible')
            close_path()
            continue

        visited[node] = True
        path.append(node)
        curr_cap += cap[node]
        curr_cost += dist[parent, node]
        parent = node

    close_path()

    return routes


def solve_colgen(data: ProblemData, subproblem: int | None, debug: bool=False):
    facilities = data.facilities

    # Limit problem (if asked)
    if subproblem is not None:
        facilities = facilities[:subproblem]

    # Use granularity
    cap = [0] + [ceil(f.material / GRANULARITY) for f in facilities]

    n_nodes = len(facilities) + 1
    # Index 0 is the starting point, index N is the "ending point" (that is the same as the starting point)
    # Cost
    dists = np.ceil(DISTANCE_GRANULARITY * points_to_distance_matrix(
        [data.distribution_center] + [p.location for p in facilities]
    ))
    max_cap = floor(MAX_CAPACITY_PER_VEHICLE / GRANULARITY)

    cgdata = CGProblemData(
        n_nodes,
        dists,
        cap,
        max_cap
    )
    ctx = Context(cgdata, debug)

    routes = initial_sol(ctx)
    used = derive_clients_count(routes, n_nodes)
    print("Initial routes:")
    print(routes)
    initial_dist = sum(r.cost for r in routes)
    print(f"Initial dist: {initial_dist / DISTANCE_GRANULARITY}")

    m = Model('vrpc')
    m.setParam("LogToConsole", 0)

    # N. of routes
    xd = m.addVar(name='Xd', vtype=GRB.CONTINUOUS)
    # Distance traveled
    xc = m.addVar(name='Xc', vtype=GRB.CONTINUOUS)
    # x[r]=1 if route k is used
    x = [m.addVar(name=r.var_name, vtype=GRB.CONTINUOUS) for r in routes]

    # Minimize sum of route lengths
    m.setObjective(qsum(routes[r].cost*x[r] for r in range(len(routes))), GRB.MINIMIZE)
    # Each client statisfied at least once
    m.addConstrs(qsum(x[r]*used[r, i + 1] for r in range(len(routes))) >= 1 for i in range(n_nodes - 1))
    # N. of routes
    m.addConstr(qsum(x[r] for r in range(len(routes))) == xd)
    # Distance traveled
    m.addConstr(qsum(x[r]*routes[r].cost for r in range(len(routes))) == xc)

    m.write('model.lp')

    # Optimize + subproblem until an optimal solution is found
    optimize_problem(ctx, m, {})

    print(f"Relaxed problem solved, cost: {m.objVal / DISTANCE_GRANULARITY}")

    # Remove relaxations and optimally solve the problem.
    branchnprice(ctx, m, initial_dist + 1)
    if ctx.best_sol is None:
        print("No feasible solution found, should be impossible")
    else:
        m = ctx.best_sol.model

        x = m.getVars()[2:]
        print(ctx.best_sol.history)
        print(f"Problem solved, cost: {m.objVal / DISTANCE_GRANULARITY}")
        print(f"Paths:")
        for v in x:
            if v.x > 0.00001:
                print(v.x, ctx.routes[v.VarName].path, v.VarName)

    if debug:
        debug_explore(ctx)
