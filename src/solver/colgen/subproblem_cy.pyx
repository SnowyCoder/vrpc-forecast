from typing import List
import numpy as np
cimport numpy as np

from math import inf

from .data import Route

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

def solve_subproblem(ctx, double[:,:] reduced_cost) -> List[Route]:
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
    cdef long[:] cap = np.array(data.cap)
    cdef double[:,:] original_dist = data.dists
    cdef int n_nodes = reduced_cost.shape[0]

    cdef int max_cap = data.max_cap
    cdef int csum = sum(data.cap)
    if csum < data.max_cap:
        max_cap = csum

    cdef double[:,:] best_cost = np.full((n_nodes, max_cap + 1), inf)
    cdef int[:,:,:] predecessor = np.full((n_nodes, max_cap + 1, 2), -1, dtype=np.int32)

    # Root
    best_cost[0, 0] = 0
    predecessor[0, 0, 0] = 0
    predecessor[0, 0, 1] = 0

    # We cannot define this inside of the cycles
    cdef double mval = inf
    cdef int mpred_i = -1
    cdef int mpred_q = -1
    cdef int qi = 0
    cdef int pj = 0
    cdef int pq = 0
    cdef int i = 0
    cdef int j
    cdef int qp
    cdef double pc = 0

    for q in range(max_cap + 1):
        # if q % 100 == 0:
        #     print(f'{q}..', end='', flush=True)
        for i in range(1, n_nodes):
            mval = inf
            mpred_i = -1
            mpred_q = -1
            qi = data.cap[i]
            for j in range(n_nodes):
                if i == j:
                    continue
                for qp in range(q - qi + 1):
                    pj = predecessor[j, qp, 0]
                    pq = predecessor[j, qp, 1]
                    # Remove 2 and 3 cycles
                    if pj == i or predecessor[pj, pq, 0] == i:
                        continue
                    pc = best_cost[j, qp] + reduced_cost[j, i]
                    if pc < mval:
                        mval = pc
                        mpred_i = j
                        mpred_q = qp

            best_cost[i, q] = mval
            predecessor[i, q, 0] = mpred_i
            predecessor[i, q, 1] = mpred_q

    # Now we can extract the paths from the last capacity level.
    chosen = [i for i in range(n_nodes) if (best_cost[i, max_cap - 1] + reduced_cost[i, 0]) < -0.001]
    routes = []

    # print([x[0] for x in chosen])

    for r in chosen:
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
