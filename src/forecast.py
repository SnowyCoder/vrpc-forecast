import datetime
from typing import Any, Dict, NamedTuple, Optional, Tuple
import pandas as pd
import logging
import random
from tqdm.contrib.concurrent import process_map
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from common import DATA_DIR, NEW_DATA_DIR, OUT_DIR

# TOTAL_WEEKS = 145
TESTING_WEEKS = 1


class DataCollection(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame


def load_data():
    print("Loading data...")
    data = aggregate_data(pd.read_csv(DATA_DIR / 'train.csv'))
    # 145 weeks, let's keep 10 for testing
    train = data[:-TESTING_WEEKS]
    test = data[-TESTING_WEEKS:]
    print(f'Data loaded\nTrain:\t{len(train)}\nTest:\t{len(test)}')

    return DataCollection(train, test)


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[['week', 'center_id', 'meal_id', 'num_orders']]
    # Rows: 1 per week
    # Columns: one per each center_id-meal_id pair
    # Cells: sum of num_orders
    df = df.pivot(index="week", columns=['center_id', 'meal_id'], values='num_orders')
    df = df.fillna(0)
    df.index.name = 'ds'
    return df


def collapse_center_meal(data: pd.DataFrame) -> pd.DataFrame:
    # def column_rename(name):
    #     if name == 'ds':
    #         return name
    #     return f"{name[0]}-{name[1]}"
    # return data.rename(columns=column_rename)

    data.columns = [
        f'{col[0]}-{col[1]}' if col != 'ds' else col for col in data.columns.values
    ]
    return data


def forecast_prophet_work(data: Tuple[pd.DataFrame, Any]) -> pd.DataFrame:
    from prophet import Prophet

    df, col = data

    subdf = df[col] # type: pd.Series
    subdf = subdf.to_frame('y').reset_index()
    # reset_index -> from { ds -> y } to {ds, y}
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    m.fit(subdf)
    future = m.make_future_dataframe(periods=TESTING_WEEKS, freq='W', include_history=False)
    forecast = m.predict(future)[['ds', 'yhat']]
    forecast['yhat']  = forecast['yhat'].abs()  # Sometimes it goes negative, and we don't wont that
    forecast = forecast.rename(columns={'yhat': col})
    return forecast


def forecast_prophet(data: DataCollection, use_cache: bool = True) -> pd.DataFrame:
    cache = load_forecast_cache('prophet')
    if cache is not None and use_cache:
        print('[prophet] Using cache')
        return cache

    # Cache prophet loading before we multiprocess
    import prophet

    df = data.train
    START = datetime.date(year=2015, month=1, day=5) # Monday
    df = df.rename(index=lambda x: START + datetime.timedelta(weeks=x))

    # SHUTUP
    logging.getLogger("cmdstanpy").disabled = True

    print("[prophet] Threaded forecasting on each data-series!")
    # Multiprocessing!!
    # On my PC this will crunch the data in a little more than 1 minute :3
    # (this would be s much faster without Python's GIL)
    mapped = process_map(forecast_prophet_work, [(df, x) for x in df.columns], chunksize=10)

    print("[prophet] Merging results")
    merged = pd.concat([mapped[0]] + [x[x.columns[1]] for x in mapped[1:]], axis=1)

    # Renormalize time format
    def weeks_between(a: datetime.date, b: datetime.date) -> int:
        # https://stackoverflow.com/a/14191915
        ma = (a - datetime.timedelta(days=a.weekday()))
        mb = (b - datetime.timedelta(days=b.weekday()))
        return (mb - ma).days // 7

    merged['ds'] = merged['ds'].map(lambda x: weeks_between(START, x.date()))

    # Renormalize column names so we can dump them to files
    merged = collapse_center_meal(merged)
    print('[prophet] Done')

    save_forecast_cache('prophet', merged)
    return merged


def forecast_average(data: DataCollection, use_cache: bool = True) -> pd.DataFrame:
    cache = load_forecast_cache('average')
    if cache is not None and use_cache:
        print('[average] Using cache')
        return cache

    WEEK_AVERAGED_COUNT = 3
    print(f'[average] Projecting last {WEEK_AVERAGED_COUNT} weeks average on next {TESTING_WEEKS} weeks')
    df = data.train
    last_week = df.index[-1]
    d = {
        'ds': [last_week + x for x in range(TESTING_WEEKS)]
    }
    for c in df.columns:
        x = df[c].iloc[-WEEK_AVERAGED_COUNT:].mean()

        d[f"{c[0]}-{c[1]}"] = [x] * TESTING_WEEKS

    res = pd.DataFrame(data=d)
    print('[average] Done')

    save_forecast_cache('average', res)
    return res


def forecast_noop(data: DataCollection) -> pd.DataFrame:
    df = data.train
    last_week = df.index[-1]
    d = {
        'ds': [last_week + x for x in range(TESTING_WEEKS)]
    }
    for c in df.columns:
        x = df[c].iloc[-1]

        d[f"{c[0]}-{c[1]}"] = [x]

    res = pd.DataFrame(data=d)
    return res


def compare_forecasts(data: DataCollection, res: Dict[str, pd.DataFrame]):
    ground_truth = collapse_center_meal(data.test.reset_index(drop=True))

    col = ground_truth.columns
    rng = random.Random('compare_forecasts.001')
    sample = rng.choices(range(len(ground_truth.columns)), k=9)
    sample = [col[i] for i in sample]
    fig, axs = plt.subplots(3, 3)

    mse_by_forecast = {}

    for i, c in enumerate(sample):
        axs[i % 3, i // 3].plot(ground_truth[c].index, ground_truth[c].values, 'o', label="ground_truth")

    for i, key in enumerate(res):
        df = res[key]
        # breakpoint()
        df = df.drop(columns=['ds'])
        # plt.plot(df[col].index, df[col].values, 'o', label=key)
        for i, c in enumerate(sample):
            axs[i % 3, i // 3].plot(df[c].index, df[c].values, 'o', label=key)
        mse = mean_squared_error(ground_truth, df)
        mse_by_forecast[key] = mse
        print(f'[compare]: {key}:\t{mse}')


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, bbox_to_anchor = (0, -0.05, 1, 1))
    text = 'MSE [' + (', '.join(f'{k}: {v:0.2f}' for k, v in mse_by_forecast.items()) + ']')
    plt.figtext(0.5, 0.01, text, horizontalalignment='center', fontsize=15)
    plt.show()


def forecast_to_weight(df: pd.DataFrame) -> Dict[int, float]:
    ingredients = pd.read_csv(NEW_DATA_DIR / 'meal_ingredients.csv')
    # Sum up all of the ingredient weights
    ing = ingredients.groupby('meal_id').sum(True)

    res = {}
    for (name, amount) in df.iloc[-1][1:].items():
        center, meal = map(int, name.split('-'))
        weight_per_unit = ing.loc[[meal]].iloc[0,0]
        old = res.get(center, 0.0)
        res[center] = old + weight_per_unit * amount
    return res


def load_forecast_cache(name) -> Optional[pd.DataFrame]:
    file = OUT_DIR / f'forecast_{name}.csv'
    if not file.is_file():
        return None
    return pd.read_csv(file)


def save_forecast_cache(name, df: pd.DataFrame):
    file = OUT_DIR / f'forecast_{name}.csv'
    df.to_csv(file, index=False)
