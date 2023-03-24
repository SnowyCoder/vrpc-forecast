import time
from collections import defaultdict

from gurobipy import Model, Var, GRB
from typing import List
from math import ceil

from .data import Context, Node
from .optimize import optimize_problem


def branchnprice(ctx: Context, m: Model, upper_bound: int | None):
    root = Node(ctx.is_debug, m, 0, {}, '')
    ctx.root = root
    ctx.push([root])
    if upper_bound is not None:
        ctx.upper_bound = upper_bound
    last_print = time.time()
    start_time = last_print

    print(f"Starting branch and price algorithm, direction: {ctx.explore_dir}")
    while (current := ctx.pop()) is not None:
        if current.model.objVal >= ctx.upper_bound:
            continue  # Pruning
        now = time.time()
        if now >= last_print + 10:
            lb = ctx.lower_bound()
            print(f"{ctx.explored_nodes}/{ctx.unexplored_nodes} [{int(lb)}/{int(current.model.objVal)}/{int(ctx.upper_bound)}] d={current.depth} t={int(now-start_time)}s")
            last_print += 10
        step(ctx, current)


def debug_explore(ctx: Context):
    path = [ctx.root]
    active = True
    while active:
        current = path[-1]
        print("------------------------------------")
        print(current.history)
        print(f'Depth: {len(path)-1}')
        for i, n in enumerate(current.children):
            print(f'{i}: {n.history}')

        if current.model.status == GRB.OPTIMAL:
            print("Routes:")
            for x in current.model.getVars()[2:]:
                if x.x > 0.001:
                    route = ctx.routes[x.VarName]
                    print(x.x, route.path, route.cost, route.has_cycles)

        while True:
            try:
                choice = input('> ')
                if choice in ['w', 'write']:
                    current.model.write('model.lp')
                    continue
                if choice in ['e', 'exit', 'q', 'quit', '-2']:
                    active = False
                    break
                if choice == 'solve':
                    optimize_problem(ctx, current.model, current.fixed, True)
                    if current.model.status == GRB.OPTIMAL:
                        print("Obj: ", current.model.objVal)
                    else:
                        print("Infeasible")
                    continue
                choice = int(choice)
                if choice == -1:
                    path = path[:-1]
                    break
                if 0 <= choice < len(current.children):
                    path.append(current.children[choice])
                    break
            except ValueError:
                pass


def step(ctx: Context, node: Node):
    children = branch(ctx, node)

    if children is None:
        # Error occurred
        return
    if len(children) == 0:
        # Solution is integer, no need to branch
        ctx.on_integer_solution(node)

    to_explore = []
    for n in children:
        optimize_problem(ctx, n.model, n.fixed)
        feasible = n.model.status == GRB.Status.OPTIMAL
        ub_pass = feasible and n.model.objVal < ctx.upper_bound
        if ctx.is_debug:
            if not feasible:
                n.history += ' - INFEASIBLE'
            elif not ub_pass:
                n.history += f' - UB-CUT ({n.model.objVal})'
            else:
                n.history += f' - ({n.model.objVal})'
        if feasible and ub_pass:
            to_explore.append(n)
    ctx.push(to_explore)


def branch(ctx: Context, node: Node) -> List[Node] | None:
    mvars = node.model.getVars()  # type: List[Var]

    xd = mvars[0]
    xc = mvars[1]
    x = mvars[2:]

    xdx = xd.x  # Uhhh, it does not carry though copies I think?
    if xdx - int(xdx) > 0.001 and abs(xdx - round(xdx)) > 0.00001:
        # Branch on n. of vehicles
        # print(f"cut vehicles {xdx}")
        ctx.branches['vehicles'] += 1
        n1 = node.child(f'xd<={int(xdx)}')
        n2 = node.child(f'xd>={int(xdx)+1}', True)

        m1 = n1.model
        m2 = n2.model
        xd = m1.getVars()[0]
        m1.addConstr(xd <= int(xdx))

        xd = m2.getVars()[0]
        m2.addConstr(xd >= int(xdx) + 1)
        return [n1, n2]

    xcx = xc.x
    if xcx - int(xcx) > 0.001 and abs(xcx - round(xcx)) > 0.00001:
        # "Branch" on total distance traveled
        ctx.branches['dist'] += 1
        # print(f"cut distance {xcx}")

        n = node.child(f'xc>={ceil(xcx)}', True)
        m = n.model
        xc = m.getVars()[1]
        m.addConstr(xc >= ceil(xcx))
        return [n]

    # Branch on arcs
    found = False
    # First: arcs with cycles
    score = defaultdict(int)
    for v in x:
        if not (v.x > 0.001 and ctx.routes[v.VarName].has_cycles):
            continue
        found = True
        # Find arcs incident to the same node (ex. 4 arcs for a single node) and give them a score
        # To do this we divide the algorithm in two parts, the first will check how many times each node is used in the
        # route, the second will give a score to each used edge incident to
        visit_count = [0] * ctx.data.n_nodes
        route = ctx.routes[v.VarName]
        path = route.path
        for n in path[1:-1]:
            visit_count[n] += 1

        # Second part: add score for each arg incident to nodes used more than once
        scored_arcs = set()
        # print("---- scoring ----")
        for i, j in zip(path[:-1], path[1:]):
            if (visit_count[i] > 1 or visit_count[j] > 1) and not (i, j) in scored_arcs:
                pass # print(f"{i} {j} -> {v.x}")
            if (visit_count[i] > 1 or visit_count[j] > 1) and not (i, j) in scored_arcs and not (i, j) in node.fixed:
                score[(i, j)] += v.x
                scored_arcs.add((i, j))

    if found:
        ctx.branches['cyclic'] += 1
        # Seems like we had at least one cyclic route
        selected_arc, _score = max(score.items(), key=lambda e: e[1], default=(None, None))
        if selected_arc is None:
            # This should not be possible
            print("------ UNABLE TO REMOVE CYCLE --------")
            print(score)
            print(node.fixed)
            for v in x:
                if v.x > 0.001 and ctx.routes[v.VarName].has_cycles:
                    print(v.x, ctx.routes[v.VarName].path)
            return None
        # print(f"cut cyclic {selected_arc} {_score}")

        n1, n2 = node.child(f'-{selected_arc}'), node.child(f'+{selected_arc}', True)
        n1.fix_arc(ctx, selected_arc, False)
        n2.fix_arc(ctx, selected_arc, True)
        return [n1, n2]

    # No cyclic routes have been found
    # Branch on fractional arcs
    # How? compute the total flow of each arc and select the arc with the highest flow
    # (but ignore arcs that have a total flow = 1, they are either fixed or, well, every route agrees on them).

    for v in x:
        if v.x < 0.0001:
            continue

        route = ctx.routes[v.VarName].path
        # No need to check for double-counting, cycles have been removed in the last stage
        for i, j in zip(route[:-1], route[1:]):
            score[(i, j)] += v.x

    selected_arc, ascore = max(((i, s) for i, s in score.items() if s < 0.999 and i not in node.fixed), key=lambda e: e[1], default=(None, None))

    if selected_arc is None:
        # The solution is not fractional and not cyclical, we found it
        return []

    # print(f"cut fractional {selected_arc} {ascore}")
    ctx.branches['fract_arc'] += 1
    n1, n2 = node.child(f'-{selected_arc}'), node.child(f'+{selected_arc}', True)
    n1.fix_arc(ctx, selected_arc, False)
    n2.fix_arc(ctx, selected_arc, True)
    return [n1, n2]


