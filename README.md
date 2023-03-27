# VRPC forecast
## Vehicle routing problem with capacity constraints

This is the project offered for "Algoritmi di Ottimizzazione", where I develop and solve a project that requires Vehicle Routing Problem with Capacity Constraints (VRPC).

## Project description
The project needs to predict the demand for meals sold by different fulfillment centers, then it needs to ship enough ingredients to allow the creation of meals.

The requirements are divided into two:
1. Forecast the future needs of the centers
2. Find the best route from a central depot to distribute it (using VRPC).

## Data
We used data from [Kaggle food-demand-forecasting](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting) and some custom generated data.
The list of ingredients for each meal has been generated using ChatGPT, and the location of each distribution center is generated randomly with some custom code (src/locations.py).

## Solving VRPC
The project main focus is solving the Vehicle Routing Problem with Capacity constraints, a well-known NP-hard problem. The project compares different formulations for the solutions and you can select between them with the `--method` argument when solving.
- `--method poly`: A solution that is polynomial in both variables and columns, the simplest method present
- `--method subtour-elim`: A method that has an exponential number of constraints, it should be theoretically faster than the polynomial one, but for this problem it doesn't seem to be
- `--method colgen`: A method that has an exponential number of variables and generates them while the problem is computed, this is the only one that can finish the real problem in a feasible amount of time


## Column Generation Implementation
To create the column generation implementation we partially used the paper from "Desrochers, Desrosiers and Solomon, 1992".
You can find the code in src/solver/colgen, there is also a debug mode used to explore the Branch and Price tree, you can
explore it using the `--debug` flag

## CLI Example
- `python3 src/main.py forecast`: computes the forecasts and compares them to each other
- `python3 src/main.py generate-data --plot`: generates centers locations and plots them visually, you can change the seed with `--seed`
- `python3 src/main.py solve --method poly --subproblem 5`: Generates data and solves the problem using only the first 4 locations and the polynomial solver
- `python3 src/main.py solve --method colgen --subproblem 10 --debug`: Generates data and solves the problem using the column generation solver and the debug branching tree explorer.

## Requirements
The project uses [Poetry](https://python-poetry.org/) to have a reproducible working environment.
If you have poetry you can use `poetry update && poetry shell` to install the used requirements and open an usable
virtual environment.

## Status: Delivered
This project is stable and does not seem to have any obvious bugs, the author does not take any responsability about its use nor about the generated data.
Run at your own risk.
