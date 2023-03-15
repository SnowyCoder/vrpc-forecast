from typing import NamedTuple, List, Dict, Tuple
import numpy as np
from gurobipy import Model
import heapq
from math import inf


# Colgen problem data
class CGProblemData(NamedTuple):
    n_nodes: int
    dists: np.ndarray
    cap: List[int]
    max_cap: int


class Route(NamedTuple):
    path: List[int]
    has_cycles: bool
    cost: int
    var_name: str


class Node:
    is_debug: bool
    model: Model
    depth: int
    fixed: Dict[Tuple[int, int], bool]
    history: str
    children: List['Node']

    def __init__(self, is_debug: bool, model: Model, depth: int, fixed: Dict[Tuple[int, int], bool], history: str = ''):
        self.is_debug = is_debug
        self.history = history
        self.model = model
        self.depth = depth
        self.fixed = fixed
        self.children = []

    def routes(self, ctx: 'Context'):
        for v in self.model.getVars()[2:]:
            yield ctx.routes[v.VarName]

    def child(self, add_hist: str, optimize_steal_model: bool = False) -> 'Node':
        history = ('' if self.history == '' or not self.is_debug else (self.history + ' '))
        if optimize_steal_model and not self.is_debug:
            model = self.model
            self.model = None
        else:
            model = self.model.copy()

        n = Node(self.is_debug, model, self.depth + 1, self.fixed, history + add_hist)
        if self.is_debug:
            self.children.append(n)
        return n

    def fix_arc(self, ctx: 'Context', arc: Tuple[int, int], fixed_on: bool):
        # Ah yes, I also love immutable structures, how can you tell -.-
        self.fixed = self.fixed.copy()
        self.fixed[arc] = fixed_on

        def remove_route(r: Route):
            var = self.model.getVarByName(r.var_name)
            if var is not None:
                self.model.remove(var)

        def index_default(lis: List[int], to_find: int, default: int):
            try:
                return lis.index(to_find)
            except ValueError:
                return default

        # Now we should remove from the model every route not following this directive
        i, j = arc
        if fixed_on:
            # We should remove every route that has:
            # - an arc (i, f) with f != j (another arc exiting i)
            # - an arc (f, j) with f != i (another arc entering j)
            for route in self.routes(ctx):
                index_i = index_default(route.path, i, len(route.path))
                if index_i < len(route.path) - 1 and route.path[index_i + 1] != j:
                    remove_route(route)
                    continue

                index_j = index_default(route.path, j, 0)
                if index_j > 0 and route.path[index_j - 1] != i:
                    remove_route(route)
        else:
            # We should remove every path with the arc (i, j) present
            for route in self.routes(ctx):
                try:
                    index_i = route.path.index(i)
                except ValueError:
                    continue
                if index_i < len(route.path) - 1 and route.path[index_i + 1] == j:
                    remove_route(route)

    def __lt__(self, other):
        if self.depth < other.depth:
            return True
        if self.depth > other.depth:
            return False
        return True


class Context:
    is_debug: bool
    data: CGProblemData
    explored_nodes: int
    max_depth: int

    to_explore: List[Tuple[int, Node]]
    lower_bound: float
    upper_bound: float
    next_route_id: int
    branches = {
        'vehicles': 0,
        'dist': 0,
        'cyclic': 0,
        'fract_arc': 0,
    }

    root: Node | None
    best_sol: Node | None

    routes: Dict[str, Route]

    def __init__(self, data: CGProblemData, is_debug: bool):
        self.is_debug = is_debug
        self.data = data
        self.explored_nodes = 0
        self.max_depth = 0

        self.to_explore = []
        self.upper_bound = inf
        self.next_route_id = 0

        self.routes = {}
        self.root = None
        self.best_sol = None

    def push(self, n: Node):
        heapq.heappush(self.to_explore, (n.model.objVal, n))

    def pop(self) -> Node | None:
        if len(self.to_explore) == 0:
            return None
        el = heapq.heappop(self.to_explore)
        self.explored_nodes += 1
        return el[1]

    def add_route(self, path: List[int], has_cycles: bool, cost: int) -> Route:
        name = f'r{len(self.routes)}'
        r = Route(path, has_cycles, cost, name)
        self.routes[name] = r
        return r

    def on_integer_solution(self, node: Node):
        val = node.model.objVal
        if val < self.upper_bound:
            print(f"### Integer solution found {val} ###")
            self.upper_bound = val
            self.best_sol = node