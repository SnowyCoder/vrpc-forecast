from math import inf
from typing import Dict, Tuple, List

import numpy as np
from gurobipy import Model, GRB, Constr, Column

from .data import Context, Route
from .subproblem import solve_subproblem


def optimize_problem(ctx: Context, m: Model, frozen: Dict[Tuple[int, int], bool], verbose=False):
    data = ctx.data
    n_nodes = data.n_nodes
    dists = data.dists

    rc = np.ndarray([n_nodes] * 2)
    while True:
        m.optimize()

        if m.status != GRB.OPTIMAL:
            break
        constrs = m.getConstrs()  # type: List[Constr]

        # Dual variable of Xd and Xc
        pi_d = constrs[n_nodes - 1].pi
        pi_c = constrs[n_nodes].pi

        # Dual variables for clients
        pi_i = [pi_d] + [c.pi for c in constrs[:n_nodes - 1]]

        # Compute reduced costs
        for i in range(n_nodes):
            for j in range(n_nodes):
                rc[i, j] = dists[i, j] - pi_i[i]

        for (i, j), v in frozen.items():
            if v:
                # Remove (i,*) and (*,j) arcs
                for h in range(n_nodes):
                    if h != j:
                        rc[i, h] = inf
                    if h != i:
                        rc[h, j] = inf
            else:
                rc[i, j] = inf
        if verbose:
            print("Costs", dists)
            print(f"Dual vars: {pi_d}, {pi_c}")
            print("Reduced constrs: \n", repr(pi_i))
            print("Reduced costs: \n", repr(rc))

        #print("----------- Solving Subproblem -----------")
        new_routes = solve_subproblem(ctx, rc)
        rclient_count = derive_clients_count(new_routes, n_nodes)

        if verbose:
            print("New routes added: ", new_routes)

        if len(new_routes) == 0:
            break

        for ir, r in enumerate(new_routes):
            # Objective: sum of x_path * cost of path, so we need the cost of the path
            # Columns?
            # - Client statisfaction constraint: how many times a client is passed
            # - Route length constraint: 1
            # - Distance traveled constraint: const of path
            cost = r.cost
            z = list(rclient_count[ir][1:]) + [1, cost] + [0] * (len(m.getConstrs()) - n_nodes - 1)
            m.addVar(obj=cost, vtype=GRB.CONTINUOUS, name=r.var_name, column=Column(z, m.getConstrs()))


def derive_clients_count(routes: List[Route], n_nodes: int) -> np.ndarray:
    res = np.full((len(routes), n_nodes), 0)

    for ipath, route in enumerate(routes):
        for node in route.path:
            res[ipath, node] += 1

    return res