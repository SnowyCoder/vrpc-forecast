from .common import ProblemData, points_to_distance_matrix, MAX_CAPACITY_PER_VEHICLE, MAX_VEHICLES
from gurobipy import Model, quicksum as qsum, GRB, Var
from typing import Dict, Tuple, List


def find_subtour(next: List[int]) -> List[List[int]]:
    visited = [False]*len(next)

    i = 0
    visited[0] = True
    while (i := next[i]) is not None and i != 0:
        visited[i] = True

    subgroups = []
    for i in range(len(next)):
        if visited[i] or next[i] is None:
            continue
        s = i
        group = [s]
        visited[i] = True
        while s != (i := next[i]):
            visited[i] = True
            group.append(i)
            assert len(group) < len(next)
        subgroups.append(group)

    return subgroups


def callback_create(x: Dict[Tuple[int, int, int], Var], glen: int, vlen: int):
    def subtour_elim(model: Model, where):
        if where != GRB.Callback.MIPSOL:
            return

        rx = model.cbGetSolution(x)

        # for k in rx:
        #     if rx[k] > 0.5:
        #         print(k)

        for k in range(vlen):
            subtours = find_subtour(connection_matrix_to_next_list(rx, glen, k))
            subtour = min(subtours, key=lambda x: len(x), default=None)
            if subtour is None:
                continue
            # print(f"Remove subtour {k} => {subtour}", (qsum(x[subtour[i], subtour[i+1], k] for i in range(len(subtour) - 1)) + x[subtour[-1],subtour[0],k] <= len(subtour) - 1))
            for ki in range(vlen):
                model.cbLazy(qsum(x[subtour[i], subtour[i+1], ki] for i in range(len(subtour) - 1)) + x[subtour[-1],subtour[0],ki] <= len(subtour) - 1)
                model.cbLazy(qsum(x[subtour[i+1], subtour[i], ki] for i in range(len(subtour) - 1)) + x[subtour[0], subtour[-1], ki] <= len(subtour) - 1)
            break

    return subtour_elim


def solve_subtour_elim(data: ProblemData, subproblem: int | None):
    facilities = data.facilities
    if subproblem is not None:
        facilities = facilities[:subproblem]
    # Index 0 is the starting point, index N is the "ending point" (that is the same as the starting point)
    # Cost
    c = points_to_distance_matrix(
        [data.distribution_center] + [p.location for p in facilities] + [data.distribution_center]
    )
    # Demand
    d = [0.] + [f.material for f in facilities]

    V = range(len(facilities) + 2)
    C = range(1, len(facilities) + 1)
    K = range(MAX_VEHICLES)

    m = Model()
    # x[i,j,k]=1 If vehicle k uses edge i->j
    x = {(i, j, k): m.addVar(name=f'x[{i},{j},{k}]', vtype=GRB.BINARY) for i in V for j in V for k in K}

    # Minimize sum of route lengths
    m.setObjective(qsum(c[i, j]*x[i, j, k] for i in V for j in V for k in K), GRB.MINIMIZE)

    n = len(C)
    m.addConstrs(x[i, i, k] == 0 for i in V for k in K)
    m.addConstrs(x[i, 0, k] == 0 for i in V for k in K)  # 0 is source
    m.addConstrs(x[n+1, i, k] == 0 for i in V for k in K)  # n+1 is sink

    # Serve all customers
    m.addConstrs(qsum(x[i, j, k] for j in V for k in K) == 1 for i in C)

    # Vehicle capacity
    m.addConstrs(qsum(d[i] * qsum(x[i, j, k] for j in V) for i in C) <= MAX_CAPACITY_PER_VEHICLE for k in K)

    # ~~~ Flow bounds ~~~
    # Every node should have exactly one entry path and one exit path
    n = len(C)
    m.addConstrs(qsum(x[0, j, k] for j in V) == 1 for k in K)
    m.addConstrs(qsum(x[i, n+1, k] for i in V) == 1 for k in K)
    m.addConstrs(qsum(x[i, h, k] for i in V) - qsum(x[h, j, k] for j in V) == 0
                 for h in C for k in K)

    m.write('model.lp')

    m.Params.LazyConstraints = 1
    m.optimize(callback_create(x, len(V), len(K)))

    next_node = lambda i, k: next(j for j in V if i != j and x[i, j, k].x > 0.5)

    def get_path(v: int):
        path = [0]
        i = 0
        while (i := next_node(i, v)) != n + 1:
            path.append(i)
        path.append(i)
        return path

    if m.status == GRB.Status.OPTIMAL:
        print(f"Problem solved, cost: {m.objVal}")
        for k in x:
            if x[k].x > 0.5:
                print(k)
        print(f"Paths:")
        for k in K:
            p = get_path(k)
            if len(p) > 2:
                print(get_path(k))
    else:
        print("No feasible solution found")


def connection_matrix_to_next_list(data: Dict[tuple[int, int, int], float], size: int, k: int) -> List[int | None]:
    p = [None] * size
    for i in range(size):
        for j in range(size):
            # (i,j) s.t. i -> j
            if i != j and data[i, j, k] > 0.5:
                p[i] = j
    return p
