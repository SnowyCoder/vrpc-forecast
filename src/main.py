import click
import math

import lib
from solver import FacilityData, ProblemData


@click.group()
def cli():
    pass


@cli.command()
@click.option('--no-cache', is_flag=True, show_default=True, default=False)
@click.option('--only-type', type=click.Choice(['average', 'sklearn', 'prophet'], case_sensitive=False))
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
@click.option('--seed')
@click.option('--plot', is_flag=True, show_default=True, default=False)
def generate_data(seed=None, plot: bool=False):
    locs = lib.locations.generate_center_locations(seed=seed)
    if plot:
        lib.locations.plot_locations(locs)


@cli.command()
@click.option('--loc-seed')
@click.option('--no-cache', is_flag=True, show_default=True, default=False)
@click.option('--subproblem', type=int)
@click.option('--method', type=click.Choice(['poly', 'subtour_elim', 'colgen'], case_sensitive=False), default='colgen')
@click.option('--debug', is_flag=True, show_default=True, default=False)
def solve(loc_seed=None, no_cache=False, subproblem: int | None = None, method: str = 'poly', debug: bool = False):
    use_cache = not no_cache

    forecast_data = lib.forecast.load_data()

    forecast = lib.forecast.forecast_average(forecast_data, use_cache=use_cache)

    locs = lib.locations.generate_center_locations(seed=loc_seed)

    materials = lib.forecast.forecast_to_weight(forecast)

    # Convert grams to Kilos
    materials = {k: math.ceil(m / 1000) for k, m in materials.items()}

    facilities = [
        FacilityData(str(k), locs.centers[k], materials[k]) for k in locs.centers
    ]

    data = ProblemData(facilities, locs.distribution_center, 100)

    if method == 'poly':
        lib.solver.solve_flow_poly(data, subproblem)
    elif method == 'subtour_elim':
        lib.solver.solve_subtour_elim(data, subproblem)
    elif method == 'colgen':
        lib.solver.solve_colgen(data, subproblem, debug)


if __name__ == '__main__':
    cli()
