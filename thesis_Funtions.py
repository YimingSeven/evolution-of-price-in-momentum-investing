# -*- coding: utf-8 -*-
"""
Pycharm Editor: Mr.seven
This is a temporary script file.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
# 计算复权价
def cal_fuquan_price(input_stock_data, fuquan_type='后复权'):
    """
    计算复权价
    :param input_stock_data:
    :param fuquan_type:复权类型，可以是'后复权'或者'前复权'
    :return:
    """
    # 创建空的df
    df = pd.DataFrame()

    # 计算复权收盘价
    num = {'后复权': 0, '前复权': -1}
    price1 = input_stock_data['close'].iloc[num[fuquan_type]]
    df['复权因子'] = (1.0 + input_stock_data['change']).cumprod()
    price2_change = df['复权因子'].iloc[num[fuquan_type]]
    df['收盘价_' + fuquan_type] = df['复权因子'] * (price1 / price2_change)

    # 计算复权的开盘价、最高价、最低价
    df['开盘价_' + fuquan_type] = input_stock_data['open'] / input_stock_data['close'] * df['收盘价_' + fuquan_type]
    # df['最高价_' + fuquan_type] = input_stock_data['最高价'] / input_stock_data['收盘价'] * df['收盘价_' + fuquan_type]
    # df['最低价_' + fuquan_type] = input_stock_data['最低价'] / input_stock_data['收盘价'] * df['收盘价_' + fuquan_type]
    # return df[[i + '_' + fuquan_type for i in '开盘价', '最高价', '最低价', '收盘价']]

    return df[[i + '_' + fuquan_type for i in ['开盘价', '收盘价']]]

# 生成形成期J月对应的t和t²
def add_constant_variables(df, month_j, initial_date_year=2006,initial_date_month=1):
    '''

    :param initial_date : 初始的时间
    :param df: 接收的df索引需要为DatetimeIndex
    :param month_j: 形成期j个月
    :param month_k: 持有期k个月
    :return:
    '''
    # 初始化形成期末的年份和月份
    end_date_year = initial_date_year
    end_date_month = initial_date_month + month_j - 1

    # 判断形成期是否跨年，若跨年并对年和月份进行修正
    if initial_date_month + month_j -1 >12:
        end_date_year = initial_date_year +1
        end_date_month = initial_date_month + month_j -1 - 12
    df_operated = df.loc[str(initial_date_year)+'-'+str(initial_date_month):str(end_date_year)+ '-' +str(end_date_month)].copy()
    # 构造形成期的天数，并循环
    for i in range(len(df_operated[str(initial_date_year)+'-'+str(initial_date_month):str(end_date_year)
                                                                              + '-' +str(end_date_month)])):
        df_operated.ix[i, 't'] = i+1.0
        df_operated.ix[i, 't_square'] = np.square(i+1.0)
        # print(df)
        # exit()
    # 初始化返回的需要进行操作的滚动向后一个年份、月份
    initial_date_rolling_year = initial_date_year
    initial_date_rolling_month = initial_date_month + 1
    # 对函数返回值的月份、年份进行跨年修正
    if end_date_year > initial_date_year:
        if initial_date_month == 12:
            initial_date_rolling_month = initial_date_month-12 +1
            initial_date_rolling_year = initial_date_year + 1
            return (df_operated.dropna(axis=0, how='any'), initial_date_rolling_year, initial_date_rolling_month)
        else:
            return (df_operated.dropna(axis=0, how='any'), initial_date_rolling_year, initial_date_rolling_month)
    return (df_operated.dropna(axis=0,how='any'), initial_date_rolling_year, initial_date_rolling_month)


# 计算并返回每日股价对t、t^2回归后 t^2前的系数
def ols_param(df):
    """

    :param df:
    :return: t^2前的系数
    """
    x = sm.add_constant(df[['t', 't_square']])
    y = df['adjust_price_fr']
    regr = sm.OLS(y, x)
    regr = regr.fit()
    params = regr.params
    return params[2]

# 求形成期的累计收益率
def total_yield(df ):
    '''

    :param df:
    :return: 返回形成期累计收益率
    '''
    df_format_change = (df['change']+1.0).prod()
    return df_format_change

# 选择持有期持有期数据
def holding_period(df,initial_date_year,initial_date_month,month_j,month_k, ):
    '''

    :param df:
    :param month_j:
    :param month_k:
    :return:
    '''
    # 初始化持有期末的年份和月份
    # end_hold_yearj = initial_date_year
    # end_hold_month = initial_date_month
    # 判断形成期是否跨年，若跨年并对持有期期初年和月份进行修正
    if initial_date_month + month_j -1 >= 12:
        initial_hold_year = initial_date_year +1
        initial_hold_month = initial_date_month + month_j - 12
    # elif initial_date_month + month_j -1 == 12:
    #     initial_hold_year = initial_date_year
    #     initial_hold_month = initial_date_month + month_j - 12
    else:
        initial_hold_year = initial_date_year
        initial_hold_month = initial_date_month + month_j
    # 判断形成期是否跨年，若跨年并对年和月份进行修正
    if initial_date_month + month_j + month_k - 1.0 > 12:
        end_hold_year = initial_date_year + ((initial_date_month + month_j + month_k - 1)//12)
        end_hold_month = (initial_date_month + month_k + month_j - 1) % 12
        # 如果形成期最后为12月时，做如下修正
        if end_hold_month == 0:
            end_hold_year = end_hold_year - 1
            end_hold_month = 12
    else:
        end_hold_year = initial_date_year
        end_hold_month = initial_date_month + month_k + month_j - 1
    # 按持有期开始至结束日期截取数据
    df_operated = df.loc[(str(initial_hold_year) + '-' + str(initial_hold_month)):(str(end_hold_year) + '-' + str(
        end_hold_month))].copy()
    return df_operated

# 选择上证指数在形成期月份的数据
def formation_month_index(df,month_j,initial_date_year=2006,initial_date_month=1):
    '''

    :param df:
    :param month_j:
    :param initial_date_year:
    :param initial_date_month:
    :return:
    '''
    # 初始化形成期末的年份和月份
    formation_end_date_year = initial_date_year
    formation_end_date_month = initial_date_month + month_j - 1

    # 判断形成期是否跨年，若跨年并对年和月份进行修正
    if initial_date_month + month_j - 1 > 12:
        formation_end_date_year = initial_date_year + 1
        formation_end_date_month = initial_date_month + month_j - 1 - 12

    # 选取出形成期月份内上证指数的数据
    df_operated_index = df.loc[str(initial_date_year) + '-' + str(initial_date_month):str(formation_end_date_year) + '-' + str(
        formation_end_date_month)].copy()

    # 计算上证指数形成期交易天数
    trading_days = df_operated_index['index_code'].value_counts()
    return (df_operated_index, trading_days[0])




