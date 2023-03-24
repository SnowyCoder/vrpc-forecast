import click
import math

import lib
from solver import FacilityData, ProblemData
from solver.colgen.data import ExploreDir


@click.group()
def cli():
    pass


@cli.command()
@click.option('--no-cache', is_flag=True, show_default=True, default=False,
              help='If set will ignore the cache created from the previous runs and recompute the data')
@click.option('--only-type', type=click.Choice(['average', 'prophet'], case_sensitive=False),
              help='Selects the types of forecasts to compare, by default all of them are selected. \'average\' uses a '
                   'simple average of previous weeks to forecast new ones while \'prophet\' tries to use the famous '
                   'library (of the same name) to generate forecasts')
def forecast(only_type=None, no_cache=False):
    use_cache = not no_cache

    data = lib.forecast.load_data()
    res = {}

    if only_type is None or only_type == 'prophet':
        res['prophet'] = lib.forecast.forecast_prophet(data, use_cache=use_cache)

    if only_type is None or only_type == 'average':
        res['average'] = lib.forecast.forecast_average(data, use_cache=use_cache)

    lib.forecast.compare_forecasts(data, res)


@cli.command()
@click.option('--seed', help='Seed used for the Random Generator, the default one is fixed for reproducibility')
def generate_data(seed=None):
    locs = lib.locations.generate_center_locations(seed=seed)
    lib.locations.plot_locations(locs)


@cli.command()
@click.argument('routes', )
@click.option('--seed', help='Seed used for location generation')
def plot_routes(routes: str, seed=None):
    """
    Plots a given solution into the location graph.

    ROUTES: the encoded solution, each route should be separated by '_' and each node in the route by '-'
            e.g. 1-2-5_3-4-6
    """

    routes = routes.split('_')
    routes = [list(map(int, r.split('-'))) for r in routes]
    locs = lib.locations.generate_center_locations(seed=seed)
    lib.locations.plot_solution(locs, routes)


@cli.command()
@click.option('--loc-seed', help='Seed to use for location data generation')
@click.option('--no-cache', is_flag=True, show_default=True, default=False,
              help='If set, the program will regenerate the ingredients data, ignoring cache status.')
@click.option('--forecast-method', type=click.Choice(['prophet', 'average', 'real']), default='average',
              help='Selects the forecast method to use')
@click.option('--subproblem', type=int, help='If set will limit the number of locations to account for.')
@click.option('--method', type=click.Choice(['poly', 'subtour_elim', 'colgen'], case_sensitive=False), default='colgen',
              help='Method used to solve the VRPC problem, read the full report (or the source code) to find the '
                   'difference between them, we recommend \'colgen\' to solve all but the simplest cases as it has been'
                   ' implemented specifically for this purpose.')
@click.option('--explore-dir', type=click.Choice(['lower', 'mixed', 'deeper'], case_sensitive=False), default='mixed',
              help='[colgen] Direction used in branch and price algorithm to explore the solution tree, lower always'
                   ' explores the node with minimal value, deeper descends into the tree in a DFS-like manner while'
                   ' mixed descends into the tree in a DSP but it also selects the lowest child at each step.\nWhile'
                   ' \'lower\' will theoretically find an optimal solution in minimal time, it also requires an absurd '
                   'amount of memory (due to unapplied optimizations), so we highly recommend \'mixed\' exploration.')
@click.option('--debug', is_flag=True, show_default=True, default=False,
              help='[colgen] Saves the state of the branch and price tree and presents an interactive shell for '
                   'exploration after the problem has been solved, useful for debugging purposes but consumes high '
                   'amounts of RAM.')
def solve(loc_seed=None, no_cache=False, forecast_method: str = 'average', subproblem: int | None = None, method: str = 'poly', debug: bool = False,
          explore_dir: ExploreDir = 'mixed'):
    use_cache = not no_cache

    forecast_data = lib.forecast.load_data()

    if forecast_method == 'average':
        forecast_results = lib.forecast.forecast_average(forecast_data, use_cache=use_cache)
    elif forecast_method == 'prophet':
        forecast_results = lib.forecast.forecast_prophet(forecast_data, use_cache=use_cache)
    else:
        forecast_results = lib.forecast.forecast_noop(forecast_data)

    locs = lib.locations.generate_center_locations(seed=loc_seed)

    materials = lib.forecast.forecast_to_weight(forecast_results)

    # Convert grams to Kilos
    materials = {k: math.ceil(m / 1000) for k, m in materials.items()}

    facilities = [
        FacilityData(str(k), locs.centers[k], materials[k]) for k in locs.centers
    ]

    data = ProblemData(facilities, locs.distribution_center)

    if method == 'poly':
        sol = lib.solver.solve_flow_poly(data, subproblem)
    elif method == 'subtour_elim':
        sol = lib.solver.solve_subtour_elim(data, subproblem)
    elif method == 'colgen':
        sol = lib.solver.solve_colgen(data, subproblem, debug, explore_dir)
    else:
        sol = None

    if sol is None:
        print("No feasible solution found")
    else:
        print(f"Found feasible solution, cost: {sol.total_cost}")
        print(f"Used paths: {len(sol.routes)}")
        for r in sol.routes:
            print(' '.join(data.facilities[x].name for x in r))
        print("Encoded solution: ", sol.encode(data))


if __name__ == '__main__':
    cli()
