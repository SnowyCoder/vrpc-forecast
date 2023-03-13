from typing import List
import heapq
import numpy as np
from math import inf

from .data import Route, Context


def solve_subproblem(ctx: Context, reduced_cost: np.ndarray, max_npaths: int) -> List[Route]:
    # I don't really understand the subproblem described in the paper
    # But it seems like a simple dynamic programming problem, so I'm writing my own

    # Problem: SPCC: Shortest Path with Capacity Costraint (and negative edges!)
    #   Resolved with: dynamic programming
    # Equation:
    # path_cost(node i, remaining_capacity q) =
    #   = min { path_cost(j, rq) + cost[i,j] }
    #     for all j != i and all rq <= q - cap_i
    # Also keep track of predecessors and remove 2-cycles
    data = ctx.data
    original_dist = data.dists
    n_nodes = reduced_cost.shape[0]

    max_cap = data.max_cap
    if (csum := sum(data.cap)) < data.max_cap:
        max_cap = csum

    best_cost = np.full((n_nodes, max_cap + 1), inf)
    predecessor = np.full((n_nodes, max_cap + 1, 2), -1)

    # Root
    best_cost[0, 0] = 0
    predecessor[0, 0] = (0, 0)

    for q in range(max_cap + 1):
        # if q % 100 == 0:
        #     print(f'{q}..', end='', flush=True)
        for i in range(1, n_nodes):
            mval = inf
            mpred = (-1, -1)
            qi = data.cap[i]
            for j in range(n_nodes):
                if i == j:
                    continue
                for qp in range(q - qi + 1):
                    if predecessor[j, qp][0] == i or predecessor[tuple(predecessor[j, qp])][0] == i:
                        continue
                    pc = best_cost[j, qp] + reduced_cost[j, i]
                    if pc < mval:
                        mval = pc
                        mpred = (j, qp)

            best_cost[i, q] = mval
            predecessor[i, q] = mpred

    # Now we can extract the paths from the last capacity level.
    chosen = heapq.nsmallest(max_npaths,
        [(v, i) for i in range(n_nodes) if (v := best_cost[i, max_cap - 1] + reduced_cost[i, 0]) < -0.001]
    )
    routes = []

    # print([x[0] for x in chosen])

    for (reduced_cost, r) in chosen:
        path = [0, r]
        visited = [False] * n_nodes
        has_cycle = False
        visited[r] = True

        original_cost = original_dist[0, r]
        c = max_cap
        while r != 0:
            r, c = predecessor[r, c]
            has_cycle |= visited[r]
            visited[r] = True
            original_cost += original_dist[path[-1], r]
            path.append(r)

        routes.append(ctx.add_route(path[::-1], has_cycle, original_cost))
        # print(reduced_cost, original_cost, path)

    return routes


# Possible implementation of the original subproblem? (doesn't work)
# def subproblem_pulling(rcost: np.ndarray, cap: List[int], max_cap: int):
#     n_nodes = rcost.shape[0]
#     lb = np.full((n_nodes, max_cap), inf)
#     ub = np.full((n_nodes, max_cap), -inf)
#     lb[0, 0] = 0
#     ub[0, 0] = 0
#     W = sorted([(q, j) for q in range(max_cap) for j in range(n_nodes) if j != 0 and q != 0])
#     n = 0
#     while n < len(W):
#         q, i = W[n]
#         qi = cap[i]
#         print(f"Step: {q}, {i}")
#         lv = inf
#         uv = -inf
#         for j in range(n_nodes):
#             if i == j:
#                 continue
#             print(j, q - qi)
#             for qp in range(0, q - qi + 1):
#                 clv = lb[j, qp] + rcost[i,j]
#                 cuv = ub[j, qp] + rcost[i,j]
#                 # print(f"- {clv} < {cuv}")
#                 if clv < lv:
#                     lv = clv
#                 if cuv > uv:
#                     uv = cuv
#         lv += rcost[i,j]
#         print(lv, uv)
#         lb[i, q] = lv
#         ub[i, q] = uv
#         # ???
#         # if lv == uv:
#         n += 1
