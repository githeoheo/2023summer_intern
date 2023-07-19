# Data analysis
import numpy as np

# API wrapper
from alpha_vantage.timeseries import TimeSeries


def download_data(config):
    """
    Download data from Alpha Vantage API.
    """
    ts = TimeSeries(key=config["alpha_vantage"]["key"])

    # Get data
    data, meta_data = ts.get_daily_adjusted(
        symbol=config["alpha_vantage"]["symbol"],
        outputsize=config["alpha_vantage"]["output_size"],
    )

    dates = [date for date in data.keys()]
    dates.reverse()

    close_prices = [
        float(data[date][config["alpha_vantage"]["key_adjusted_close"]])
        for date in data.keys()
    ]
    close_prices.reverse()
    close_prices = np.array(close_prices)

    num_data_points = len(dates)

    display_date_range = (
        "from {} to {}".format(dates[0], dates[num_data_points - 1])
        if num_data_points > 1
        else dates[0]
    )

    print(f"Downloaded {num_data_points} data points {display_date_range}.")

    return dates, close_prices, num_data_points, display_date_range


def prepare_x(x, window_size):
    """
    Prepare data for training.
    """
    output = np.lib.stride_tricks.as_strided(
        x,
        shape=(x.shape[0] - window_size + 1, window_size),
        strides=(x.strides[0], x.strides[0]),
    )
    return output[:-1], output[-1]


def prepare_y(x, window_size):
    """
    Prepare data for training.
    """
    return x[window_size:]
