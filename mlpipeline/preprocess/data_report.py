# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 23/12/2019

Description:
    Generate data report for EDA
"""
import sys
import os

from docx import Document
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pyecharts import Pie, Bar, Line
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sys.path.append('../../')
import conf
from utils.utils import read_data_dir

# global default variables
FONT = fm.FontProperties(fname=os.path.join(conf.lib_dir, 'simsun.ttc'))
DEFAULT_PIC_WIDTH = 4
DEFAULT_MAX_CATE = 15
OUTPUT_PATH = conf.figs_dir


def __plot_chart(df, chart_name, cate_col, targe_col, output_path, chart_type="Pie", use_sort=True):
    if use_sort:
        df = df.sort_values(targe_col, ascending=False)
    chart = eval(chart_type)(chart_name + "(Top%s)" % DEFAULT_MAX_CATE) if chart_type != "Line" else Line(chart_name)
    if chart_type == 'Bar':
        chart.add("", df[cate_col].values[0:DEFAULT_MAX_CATE], df[targe_col].values[0:DEFAULT_MAX_CATE],
                  mark_line=["average"], mark_point=["max", "min"], is_legend_show=True)
    elif chart_type == 'Pie':
        chart.add("", df[cate_col].values[0:DEFAULT_MAX_CATE], df[targe_col].values[0:DEFAULT_MAX_CATE],
                  is_label_show=True, is_legend_show=False)
    else:
        chart.add("", df[cate_col].values, df[targe_col].values,
                  is_label_show=True, is_legend_show=False)
    chart.render(output_path)


def __pandas_plot_chart(df, title, kind='line', logx=False, logy=False):
    ax = df.plot(kind=kind, logx=logx, logy=logy, legend=True)
    ax.set_title(title, fontproperties=FONT)
    plt.legend(prop=FONT)
    fig = ax.get_figure()
    fig.savefig(os.path.join(OUTPUT_PATH, title))


def device_info_report(device_info_df, weather_df, meter_reading_df, building_info_df, freq=30, lags=25):
    assert not (device_info_df.empty and weather_df.empty and meter_reading_df.empty), \
        'input dfs should not be empty'

    device_category_count = device_info_df.groupby("device_category", as_index=False)[['device_id']].count()
    __plot_chart(device_category_count, '设备类型分布图', 'device_category', 'device_id',
                 os.path.join(conf.figs_dir, "device_category_pie.png"))

    device_building_df = device_info_df.merge(building_info_df, on=['building_id'], how='left')
    building_device_count = device_building_df.groupby("building_name", as_index=False)[['device_id']].count()
    __plot_chart(building_device_count, '建筑设备数分布图', 'building_name', 'device_id',
                 os.path.join(conf.figs_dir, "building_num_device_pie.png"))

    building_device_count = device_building_df.groupby("building_name", as_index=False)[['device_category']].count()
    __plot_chart(building_device_count, '建筑设备类型分布图', 'building_name', 'device_category',
                 os.path.join(conf.figs_dir, "building_num_device_cate_pie.png"))

    # TODO: 添加设备与天气关系
    # TODO: 添加设备与工作日以及节假日的关系

    # # plot the each device tendency in each building
    # meter_device_df = meter_reading_df.merge(device_info_df, on=['device_id', 'building_id'], how='left')
    # for device in set(meter_device_df['device_name']):
    #     dfs = []
    #
    #     for building in set(meter_device_df['building_id']):
    #         sub_df = meter_device_df[(meter_device_df.device_name == device) &
    #                                  (meter_device_df.building_id == building)][['date', 'meter_reading']]
    #
    #         if not sub_df.empty:
    #             building_name = building_info_df[building_info_df.building_id == building]['building_name'].iloc[0]
    #             sub_df.rename(columns={'meter_reading': building_name}, inplace=True)
    #             sub_df.set_index('date', inplace=True)
    #             dfs += [sub_df]
    #
    #     df = pd.concat(dfs, axis=1)
    #     __pandas_plot_chart(df, title='%s趋势图' % device)

    # # moving average method for trend, season and residual decompose
    # # TODO: 用sota的方法进行趋势,季节项分解
    # decomposed_meter_reading = sm.tsa.seasonal_decompose(meter_reading_df[['meter_reading','date']].set_index('date', inplace=False),freq=freq)
    # fig = decomposed_meter_reading.plot()
    # fig.savefig(os.path.join(conf.figs_dir,'趋势季节分解图'))
    #
    # acf_fig = plot_acf()
    # pacf_fig = plot_pacf()


def building_info_report(building_info_df):
    assert not building_info_df.empty, 'input df should not be empty'

    __plot_chart(building_info_df, '各建筑面积图',
                 'building_name', 'building_area',
                 os.path.join(conf.figs_dir, "building_area_bar.png"), 'Bar')

    __plot_chart(building_info_df, '各建筑居民数图',
                 'building_name', 'num_resident',
                 os.path.join(conf.figs_dir, "building_resident_bar.png"), 'Bar')

    __plot_chart(building_info_df, '各建筑楼层数图',
                 'building_name', 'building_floor',
                 os.path.join(conf.figs_dir, "building_floor_bar.png"), 'Bar')

    __plot_chart(building_info_df, '各建筑住户图',
                 'building_name', 'num_room',
                 os.path.join(conf.figs_dir, "building_room_bar.png"), 'Bar')


def generate_reports(dir_path):
    doc = Document()
    device_info_df, weather_df, meter_reading_df, building_info_df = read_data_dir(dir_path)


if __name__ == "__main__":
    pass
