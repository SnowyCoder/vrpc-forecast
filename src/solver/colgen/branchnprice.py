from collections import defaultdict

from gurobipy import Model, Var, GRB
from typing import List
from math import ceil

from .data import Context, Node
from .optimize import optimize_problem


def branchnprice(ctx: Context, m: Model):
    root = Node(m, 0, {}, '')
    ctx.root = root
    ctx.push(root)

    while (current := ctx.pop()) is not None:
        print(f"Solving {ctx.explored_nodes}/{len(ctx.to_explore)}: {current.model.objVal}")
        if current.model.objVal < ctx.upper_bound:
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
                    # path = [0, 7, 3, 1, 9, 4, 8, 5, 2, 0]
                    #
                    # def addpath(p, name):
                    #     cc = [int(i in p) for i in range(1, ctx.data.n_nodes)]
                    #     print(cc)
                    #     import gurobipy
                    #     m = current.model
                    #     z = cc + [1, 197565] + [0] * (len(m.getConstrs()) - ctx.data.n_nodes - 1)
                    #     m.addVar(obj=197565, vtype=GRB.CONTINUOUS, name=name, column=gurobipy.Column(z, m.getConstrs()))
                    # addpath(path, 'test1')
                    # addpath([0, 6, 0], 'test2')

                    optimize_problem(ctx, current.model, current.fixed, True)
                    print("Obj: ", current.model.objVal)
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

    if len(children) == 0:
        # Solution is integer, no need to branch
        ctx.on_integer_solution(node)

    for n in children:
        optimize_problem(ctx, n.model, n.fixed)
        if n.model.status != GRB.Status.OPTIMAL:
            n.history += ' - INFEASIBLE'
        elif n.model.objVal >= ctx.upper_bound:
            n.history += f' - UB-CUT ({n.model.objVal})'
        else:
            n.history += f' - ({n.model.objVal})'
            ctx.push(n)


def branch(ctx: Context, node: Node) -> List[Node]:
    mvars = node.model.getVars()  # type: List[Var]

    xd = mvars[0]
    xc = mvars[1]
    x = mvars[2:]

    xdx = xd.x  # Uhhh, it does not carry though copies I think?
    if xdx - int(xdx) > 0.001:
        # Branch on n. of vehicles
        print("cut vehicles")
        ctx.branches['vehicles'] += 1
        m1 = node.model.copy()  # type: Model
        m2 = node.model.copy()
        xd = m1.getVars()[0]
        m1.addConstr(xd <= int(xdx))

        xd = m2.getVars()[0]
        m2.addConstr(xd >= int(xdx) + 1)
        return [node.child(m1, f'xd<={int(xdx)}'), node.child(m2, f'xd>={int(xdx)+1}')]

    xcx = xc.x
    if xcx - int(xcx) > 0.001:
        # "Branch" on total distance traveled
        ctx.branches['dist'] += 1
        print(f"cut distance {xcx}")

        m = node.model.copy()
        xc = m.getVars()[1]
        m.addConstr(xc >= ceil(xcx))
        return [node.child(m, f'xc>={ceil(xcx)}')]

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
            return []
        print(f"cut cyclic {selected_arc} {_score}")

        n1, n2 = node.child(node.model.copy(), f'-{selected_arc}'), node.child(node.model.copy(), f'+{selected_arc}')
        n1.fix_arc(ctx, selected_arc, False)
        n2.fix_arc(ctx, selected_arc, True)
        return [n1, n2]

    # No cyclic routes have been found
    # Branch on fractional arcs
    # How? compute the total flow of each arc and select the arc with the highest flow
    # (but ignore arcs that have a total flow = 1, they are either fixed or, well, every route agrees on them).
    for v in x:
        if v.x < 0.001:
            continue

        route = ctx.routes[v.VarName].path
        # No need to check for double-counting, cycles have been removed in the last stage
        for i, j in zip(route[:-1], route[1:]):
            score[(i, j)] += v.x

    selected_arc, _score = max(((i, s) for i, s in score.items() if s < 0.999), key=lambda e: e[1], default=(None, None))

    if selected_arc is None:
        # The solution is not fractional and not cyclical, we found it
        return []

    print(f"cut fractional {selected_arc}")
    ctx.branches['fract_arc'] += 1
    n1, n2 = node.child(node.model.copy(), f'-{selected_arc}'), node.child(node.model.copy(), f'+{selected_arc}')
    n1.fix_arc(ctx, selected_arc, False)
    n2.fix_arc(ctx, selected_arc, True)
    return [n1, n2]


