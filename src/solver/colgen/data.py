from typing import NamedTuple, List, Dict, Tuple, Literal
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
    path: Tuple[int, ...]
    has_cycles: bool
    cost: int
    var_name: str


def _index_default(lis: Tuple[int, ...], to_find: int, start: int, end: int) -> int | None:
    try:
        return lis.index(to_find, start, end)
    except ValueError:
        return None


def _has_arc(lis: Tuple[int, ...], t: Tuple[int, int] | Tuple[int, Literal['*']] | Tuple[Literal['*'], int],
             no_sub: int | None = None):
    # when (a, *) or (*, b) -> return True if exists * != no_sub
    # when (a, b) return True
    a, b = t
    if a == 0 or b == 0:
        return False
    if type(a) != str:  # We're in either (a, *) or (a, b)
        curr_index = 1
        while True:
            ia = _index_default(lis, a, curr_index, -1)
            if ia is None:
                return False
            if lis[ia + 1] == b or (b == '*' and lis[ia + 1] != no_sub):
                return True
            curr_index = ia + 1
    if type(b) != str:  # We're in (*, b)
        # We only need to check lis[1:-1] because the list is of form [0, ... 0]
        curr_index = 1
        while True:
            ib = _index_default(lis, b, curr_index, -1)
            if ib is None:
                return False
            if lis[ib - 1] != no_sub:
                return True
            curr_index = ib + 1
    return True


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
        if self.is_debug:
            history = ('' if self.history == '' else (self.history + ' ')) + add_hist
        else:
            history = ''

        if optimize_steal_model and not self.is_debug:
            model = self.model
            self.model = None
        else:
            model = self.model.copy()

        n = Node(self.is_debug, model, self.depth + 1, self.fixed, history)
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
                # Don't remove, otherwise the model might become unfeasible (and we cannot extract the dual variables)
                var.Obj = 10000000
                # self.model.remove(var)

        # Now we should remove from the model every route not following this directive
        i, j = arc
        if fixed_on:
            # We should remove every route that has:
            # - an arc (i, f) with f != j (another arc exiting i)
            # - an arc (f, j) with f != i (another arc entering j)
            for route in self.routes(ctx):
                if _has_arc(route.path, (i, '*'), j) or _has_arc(route.path, ('*', j), i):
                    remove_route(route)
        else:
            # We should remove every path with the arc (i, j) present
            for route in self.routes(ctx):
                if _has_arc(route.path, (i, j)):
                    remove_route(route)

    def __lt__(self, other):
        if self.depth < other.depth:
            return True
        if self.depth > other.depth:
            return False
        return True


ExploreDir = Literal['lower', 'deeper', 'mixed']


class Context:
    is_debug: bool
    data: CGProblemData
    explored_nodes: int
    explore_dir: ExploreDir

    to_explore: List[Tuple[int, Node]]
    explore_stack: List[Node]  # Used with mixed exploration
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
    routes_by_path: Dict[Tuple[int, ...], Route]

    def __init__(self, data: CGProblemData, is_debug: bool, explore_dir: ExploreDir):
        self.is_debug = is_debug
        self.data = data
        self.explored_nodes = 0
        self.unexplored_nodes = 0
        self.explore_dir = explore_dir

        self.to_explore = []
        self.explore_stack = []
        self.upper_bound = inf
        self.next_route_id = 0

        self.routes = {}
        self.routes_by_path = {}
        self.root = None
        self.best_sol = None

    def lower_bound(self) -> float:
        if self.explore_dir == 'lower':
            return self.to_explore[0][0] if len(self.to_explore) else 0
        else:
            return min(map(lambda n: n.model.objVal, self.explore_stack), default=0)

    def push(self, n: List[Node]):
        n = list(filter(lambda x: x.model.objVal < self.upper_bound, n))
        self.unexplored_nodes += len(n)
        if self.explore_dir == 'lower':
            for e in n:
                heapq.heappush(self.to_explore, (e.model.objVal, e))
        elif self.explore_dir == 'deeper':
            self.explore_stack.extend(n[::-1])
        elif self.explore_dir == 'mixed':
            # Put most promising nodes first
            self.explore_stack.extend(sorted(n, key=lambda e: -e.model.objVal))

    def pop(self) -> Node | None:
        if len(self.to_explore) + len(self.explore_stack) == 0:
            return None
        self.explored_nodes += 1
        self.unexplored_nodes -= 1
        if self.explore_dir == 'lower':
            return heapq.heappop(self.to_explore)[1]
        else:
            return self.explore_stack.pop()

    def add_route(self, path: List[int], has_cycles: bool, cost: int) -> Route:
        name = f'r{len(self.routes)}'
        path = tuple(path)
        if r := self.routes_by_path.get(path, None):
            return r
        r = Route(path, has_cycles, cost, name)
        self.routes[name] = r
        self.routes_by_path[path] = r
        return r

    def on_integer_solution(self, node: Node):
        val = node.model.objVal
        if val < self.upper_bound:
            print(f"### Integer solution found {val}, explored nodes: {self.explored_nodes}, d={node.depth} ###")
            self.upper_bound = val
            self.best_sol = node
        if self.explore_dir == 'lower':
            self.to_explore.clear()
