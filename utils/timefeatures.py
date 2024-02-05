# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
import re
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


def seq2instance(data, num_of_history, num_of_predict):
    num_step, dims = data.shape
    num_sample = num_step - num_of_history - num_of_predict + 1
    x = np.zeros(shape = (num_sample, num_of_history, dims), dtype=np.float)
    y = np.zeros(shape = (num_sample, num_of_predict, dims), dtype=np.float)
    for i in range(num_sample):
        x[i] = data[i : i + num_of_history]
        y[i] = data[i + num_of_history : i + num_of_history + num_of_predict]
    return x, y

def get_freq_delta_second(freq):
    assert isinstance(freq, str)
    if 'H' in freq:
        number = re.findall("\d+", freq)
        assert len(number) == 1
        freq_delta_second = int(number[0]) * 60 * 60
    elif ('min' in freq) or ('T' in freq):
        number = re.findall("\d+", freq)
        assert len(number) == 1
        freq_delta_second = int(number[0]) * 60
    else:
        raise ValueError('freq error!')
    return freq_delta_second

def time_embedding(df, embed, features, freq):
    if embed == 'discrete':
        df.set_index("date", inplace=True)
        time = df.index
        if features == 2:
            dayofweek = np.reshape(time.weekday, newshape=(-1, 1))
            timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // freq
            timeofday = np.reshape(timeofday, newshape=(-1, 1))
            time = np.concatenate((dayofweek, timeofday), axis=-1)
        elif features == 4:
            monthofyear = np.reshape(time.month, newshape=(-1, 1))
            dayofmonth = np.reshape(time.day, newshape=(-1, 1))
            dayofweek = np.reshape(time.weekday, newshape=(-1, 1))
            timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // freq
            timeofday = np.reshape(timeofday, newshape=(-1, 1))
            time = np.concatenate((monthofyear, dayofmonth, dayofweek, timeofday), axis=-1)
    elif embed == 'continuous':
        time = time_features(pd.to_datetime(df['date'].values), freq=freq)
        time = time.transpose(1, 0)
    return time

def get_timestamp(embed, features, start_time, num_of_steps, freq, num_of_history, num_of_predict, train_ratio, val_ratio, test_ratio):

    if isinstance(start_time, str) is not True:
        start_time = str(start_time)

    freq_delta_second = get_freq_delta_second(freq)
    time_per_day = int(24 * 60 * 60 // freq_delta_second)

    t = pd.date_range(start_time, periods=num_of_steps, freq=freq)
    df = pd.DataFrame({"date": t})
    df['date'] = pd.to_datetime(df['date'])
    if embed == 'discrete':
        timestamp = time_embedding(df, embed, features, freq_delta_second)
    elif embed == 'continuous':
        timestamp = time_embedding(df, embed, features, freq)

    return timestamp, time_per_day

def convert_array2timestamp(time_array, embed, features, freq):
    assert time_array.ndim < 3
    if time_array.ndim == 1:
        np.expand_dims(time_array, axis=-1)
    num_step, seq_len = time_array.shape

    timestamp = []
    for i in range(seq_len):
        t_i = pd.to_datetime(time_array[:, i])
        df = pd.DataFrame({"date": t_i})
        if embed == 'discrete':
            freq_delta_second = get_freq_delta_second(freq)
            t_i = time_embedding(df, embed, features, freq_delta_second)
        elif embed == 'continuous':
            t_i = time_embedding(df, embed, features, freq)
        timestamp.append(t_i)
    timestamp = np.stack(timestamp, 1)
    return timestamp
