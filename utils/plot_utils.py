# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 17/12/2019

Description:
    Utility function for plot some graphs
"""
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt
# import pandas_profiling
# import chart_studio.plotly as py
# import cufflinks as cf

sys.path.append('../')
import conf

# Global setting 


def plot_dist_of_cols(dfs, plot_cols):
    """Df in dfs should be sorted by date"""
    for df in dfs:
        axes = df.boxplot(column = plot_cols, rot=90)
        title = '%s - %s'%(df['dt'].iloc[0],df['dt'].iloc[-1])
        axes.set_title(title)
        plt.show()
        save_path = os.path.join(conf.FIGURE_DIR, '%s_dist.png' % title)
        plt.savefig(save_path)
        print('%s_dist.png has been saved in %s ' % (title, save_path))

def pandas_profiling(dfs):
    """A wraper for pandas profiling"""
    pass
