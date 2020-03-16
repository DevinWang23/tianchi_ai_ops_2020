# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 23/10/2019

Description:
    Do the feature engineering

"""
import sys
import datetime

import pandas as pd
import multiprocessing
from chinese_calendar import is_holiday, is_in_lieu

sys.path.append('../')
from utils.utils import back_fill

MIN_X = -1000000
MAX_X = 1000000
MIN_Y = -1
MAX_Y = 100
SMALL_VALUE = 0.01
DEFAULT_MISSING_FLOAT = -1.234
DEFAULT_MISSING_STRING = 'U'
USE_METER_READING = True
USE_TEMP_MORNING = True
USE_TEMP_EVENING = True


def __apply_df(args):
    df, is_train, predict_targets = args
    return __create_daily_features(df, is_train=is_train, predict_targets=predict_targets)


def __add_building_feature(ret, building_info_df):
    ret = ret.merge(building_info_df, on='building_id', how='left')
    # TODO: 对数值性特征作无纲量化
    current_year = datetime.datetime.now().year
    ret['completion_years'] = ret['year_built'].apply(lambda x: current_year - int(x))
    ret.drop(columns=['year_built'], inplace=True)
    return ret


def __add_weather_feature(meter_reading_df, weather_df):
    weather_df.drop_duplicates(inplace=True)
    meter_weather_df = meter_reading_df.merge(weather_df, on=['date'], how='left')
    return meter_weather_df


def __add_device_feature(ret, device_info_df):
    ret = ret.merge(device_info_df, on=['device_id', 'building_id'], how='left')
    return ret


def __create_date_feature(ret):
    """day_of_week, holiday, day_of_month, month_of_year"""
    if 'date' not in set(ret.columns):
        raise KeyError('date should be a column in input dataframe')

    ret['date'] = pd.to_datetime(ret['date'])
    ret['day_of_month'] = ret['date'].dt.day
    ret['day_of_week'] = ret['date'].dt.weekday
    ret['month_of_year'] = ret['date'].dt.month
    ret['is_holiday'] = ret['date'].apply(lambda x: 1 if is_holiday(x) else 0)
    ret['is_break'] = ret['date'].apply(lambda x: 1 if is_in_lieu(x) else 0)  # 是否调休
    # TODO: 加入是否为月中,月初以及月末特征
    return ret


def __create_skew_kurt_cv(df, cols, window_list, window_size, min_periods, start_index):
    """
    create statistics features about distribution
    :param df:
    :param cols:
    :param window_list:
    :param window_size:
    :param min_periods:
    :param start_index:
    :return:
    """
    dfs = []
    for i_window in window_list:
        target_df = df[cols].iloc[start_index * i_window * window_size:]
        df_skew = target_df.rolling(i_window * window_size, min_periods=min_periods).skew().iloc[start_index:]
        df_skew.columns = [x + "_skew_" + str(i_window * window_size) for x in df_skew.columns]
        dfs.append(df_skew)
        df_kurt = target_df.rolling(i_window * window_size, min_periods=min_periods).kurt().iloc[start_index:]
        df_kurt.columns = [x + "_kurt_" + str(i_window * window_size) for x in df_kurt.columns]
        dfs.append(df_kurt)
        df_cv = (target_df.rolling(i_window * window_size, min_periods=min_periods).std() /
                 target_df.rolling(i_window * window_size, min_periods=min_periods).mean()).iloc[start_index:]
        df_cv.columns = [x + "_cv_" + str(i_window * window_size) for x in df_cv.columns]
        dfs.append(df_cv)
    return dfs


def __create_daily_features(df, window_list=[1, 2, 7], window_size=7,
                            min_periods=1, predict_targets=[1, 2, 7], is_train=True,
                            normalize=True):
    """
    create min, max, mean for different sliding window size
    :param df:
    :param window_list:
    :param window_size:
    :param min_periods:
    :param predict_target:
    :param is_train:
    :param normalize:
    :return:
    """
    df = back_fill(df, freq='D')
    start_index = 0 if is_train else -1  # for judging train from prediction
    df_index = df[['device_id', 'building_id', 'date']].iloc[start_index:]
    df_cate = df[['city', 'weather_morning', 'weather_evening']].iloc[start_index:]
    dfs, cols = [df_index, df_cate], []
    [cols.append(col) for bool_judge, col in [(USE_METER_READING, 'meter_reading'), (USE_TEMP_MORNING, 'temp_morning'),
                                              (USE_TEMP_EVENING, 'temp_evening')]
     if bool_judge and col in df.columns]
    df_cols = df[cols]
    target_df = df_cols.tail(window_list[-1] * window_size) if not is_train else df_cols

    # normalize features by mean
    tmp_min_periods = min_periods if is_train else min(window_size, target_df.shape[0])
    mean_data = target_df.rolling(window_list[0] * window_size, min_periods=tmp_min_periods).mean().iloc[start_index:]
    mean_data = mean_data.where(mean_data > 0.01).fillna(1.0)
    if not normalize:
        for col in ['meter_reading', 'temp_morning', 'temp_evening']:
            if col in cols:
                mean_data.loc[:, col] = 1.0
    else:
        for col in ['meter_reading', 'temp_morning', 'temp_evening']:
            if col in cols:
                mean_data.loc[:, col] = mean_data[col].apply(lambda x: max(0.2, x))
    dfs.append(mean_data)

    for i_window in window_list:
        target_df = df_cols
        dfs.append((target_df.rolling(i_window * window_size, min_periods=tmp_min_periods).min().iloc[start_index:]
                    / mean_data).rename(
            columns=dict(zip(cols, [s + "_min_%s" % (i_window * window_size) for s in cols]))))
        dfs.append((target_df.rolling(i_window * window_size, min_periods=tmp_min_periods).max().iloc[start_index:]
                    / mean_data).rename(
            columns=dict(zip(cols, [s + "_max_%s" % (i_window * window_size) for s in cols]))))
        dfs.append((target_df.rolling(i_window * window_size, min_periods=tmp_min_periods).std().iloc[start_index:]
                    / mean_data).rename(
            columns=dict(zip(cols, [s + "_std_%s" % (i_window * window_size) for s in cols]))))
        dfs.append((target_df.rolling(i_window * window_size, min_periods=tmp_min_periods).mean().iloc[start_index:]
                    / mean_data).rename(
            columns=dict(zip(cols, [s + "_mean_%s" % (i_window * window_size) for s in cols]))))

    # the last period meter reading
    min_window = window_list[0]
    for target in predict_targets:
        shift_length = min_window * window_size - target
        if shift_length > 0 and target < 7:
            dfs.append((df_cols[['meter_reading']].shift(shift_length).iloc[start_index:] /
                        mean_data[['meter_reading']]).rename(
                columns=dict({'meter_reading': 'meter_reading_last_period_for_%s' % target})))

    # @TODO: 样本历史数据够多的情况下加入去年同期读数
    dfs.append((df_cols.iloc[start_index:] / mean_data).rename(columns=dict(zip(cols, [s + "_cur" for s in cols]))))
    observe_window = min_periods if is_train else min(df_cols.shape[0], window_size * window_list[-1])
    dfs += __create_skew_kurt_cv(df_cols, cols, [window_list[-1]], window_size=window_size, min_periods=observe_window,
                                 start_index=start_index)
    xdf = pd.concat(dfs, axis=1).fillna(DEFAULT_MISSING_FLOAT)

    if is_train:
        dfs_y = []
        for i in predict_targets:
            tmp = (df_cols[['meter_reading']].rolling(i, min_periods=min_periods).mean().shift(-i) / mean_data[
                ['meter_reading']]).rename(
                columns=dict({"meter_reading": "y_%s" % i}))
            dfs_y.append(tmp)
        label = pd.concat(dfs_y, axis=1).clip(MIN_Y, MAX_Y)
        training_data = pd.concat([xdf, label], axis=1)
    else:
        training_data = xdf

    return training_data


def feature_engineering_pandas(meter_reading_df, building_info_df, device_info_df, weather_df, logger,
                               predict_targets=[1, 2, 7], start_date=None, end_date=None, is_train=True):
    meter_reading_df = meter_reading_df[meter_reading_df['date'] >= start_date] if start_date is not None else \
        meter_reading_df
    meter_reading_df = meter_reading_df[meter_reading_df['date'] <= end_date] if end_date is not None else \
        meter_reading_df
    meter_weather_df = __add_weather_feature(meter_reading_df, weather_df)

    # create the sliding window features with parallel
    logger.info('开始构造滑窗特征')
    sub_dfs = dict(tuple(meter_weather_df.groupby(['building_id', 'device_id'])))
    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1))
    result = pool.map_async(__apply_df, [(sub_dfs[key], is_train, predict_targets) for key in sub_dfs.keys()])
    pool.close()
    ret = pd.concat(list(result.get())).fillna(DEFAULT_MISSING_FLOAT)
    logger.info('构造滑窗特征后，当前维度为%s' % ret.shape[1])

    logger.info('开始构造日期特征')
    ret = __create_date_feature(ret)
    logger.info('构建日期特征后,当前维度为%s' % ret.shape[1])
    if building_info_df is not None:
        logger.info('开始构造楼宇特征')
        ret = __add_building_feature(ret, building_info_df)
        logger.info('构建楼宇特征后,当前维度为%s' % ret.shape[1])
    if device_info_df is not None:
        logger.info('开始构造设备特征')
        ret = __add_device_feature(ret, device_info_df)
        logger.info('构建设备特征后,当前维度为%s' % ret.shape[1])

    return ret
