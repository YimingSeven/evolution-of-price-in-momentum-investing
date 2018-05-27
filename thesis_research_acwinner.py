# -*- coding: utf-8 -*-
"""
Pycharm Editor: Mr.seven
This is a temporary script file.
"""
import pandas as pd
import statsmodels.api as sm
import os
import numpy as np
import time
from thesis_project import thesis_Funtions
# 当列太多时不换行
pd.set_option('expand_frame_repr',False)

# 设置开始计时时间
start_time = time.time()
# ============================初始化部分============================
df_temporary = pd.DataFrame()
df_temporary_params = pd.DataFrame()
df_temporary_end = pd.DataFrame()
df_formation = pd.DataFrame()
df_formation_origin = pd.DataFrame()
stock_list=[]    #

t_square_params_list,stock_code_list, formation_total_return = list(),list(),list()# 创建两个列表

# ============================初始化参数============================
initial_date_year = 2014
initial_date_month = 7
month_j= 6
month_k= 24


# ============================读取上证指数数据============================
index_data = pd.read_csv('E://pyseven//thesis_project//thesis_data//index_data//index000001.csv',index_col='date',parse_dates=True)
index_data.sort_index(inplace=True)
formation_index, formation_index_trading_days = thesis_Funtions.formation_month_index(index_data,month_j=month_j,
                                                                        initial_date_year=initial_date_year,initial_date_month=initial_date_month)
# print(formation_index)
# print(formation_trading_days)
# exit()

# ============================读取股票数据============================
#遍历导入数据
for roots, dirs, files in os.walk('E://pyseven//thesis_project//thesis_data//stock_data'):
    if files :
        for i in files:
            if i.endswith('.csv'):
                stock_list.append(i)

for file_name in stock_list:
    df = pd.read_csv('E://pyseven//thesis_project//thesis_data//stock_data//'+file_name , encoding='gbk')
    # df.columns = [i.encode('utf8') for i in df.columns]
    df=df[['code','date','open','close','change','traded_market_value']]
    df['date']=pd.to_datetime(df['date'])
    # df.rename(columns={'adjust_price_f':'adjust_price_fr'},inplace=True)
    df.sort_values(by='date', inplace=True)
    # 计算复权价
    df[['adjust_open','adjust_price_fr']] = thesis_Funtions.cal_fuquan_price(df, fuquan_type='前复权')
    # 判断每天开盘是否涨停
    df.loc[df['adjust_open'] > df['adjust_price_fr'].shift(1) *1.097, 'limit_up'] = 1
    df['limit_up'].fillna(0, inplace = True)
    # 选取2006/01/01后的数据
    df = df.loc[(df['date'] >= '2005/12/31') & (df['date'] <= '2016/12/31')]

    # df_temporary 2006/1/1至2016/12/31的单只股票数据
    df_temporary = df
    # break  #先实验单只股票，后续对多只股票
    df_temporary.reset_index(drop=True,inplace=True)
    # 将date设置为index
    df_temporary.set_index('date' ,inplace=True)
    # df_temporary.index = pd.DatetimeIndex(df_temporary['date'])
    # 形成期3个月，每日股价对t、t^2回归
    # 用df_temporary.axes来查看是否index变为DatetimeIndex形式

    # ============================设置形成期，对t、t^2回归并得到变量前系数============================
    try:
        # df1_t_square是形成期的有t、t^2的单只股票数据
        df1_t_square, year, month = thesis_Funtions.add_constant_variables(df_temporary, month_j=month_j, initial_date_year=initial_date_year,
                                                                      initial_date_month =initial_date_month)
        # 将形成期内股票的前复权数据中的nan值设置成前一个交易日的值，若没有则设为0
        df1_t_square['adjust_price_fr'].fillna(method='ffill',inplace =True)
        df1_t_square['adjust_price_fr'].fillna(value=0,inplace=True)
        # 统计每只股票在排名期的交易日天数
        formation_stock_trading_days = df1_t_square['code'].value_counts()

        # 剔除在排名期内累计停牌超过（10*月数）天的股票，即如果排名期为3个月，就剔除累计停牌超过30天的股票
        trading_stock_list = formation_stock_trading_days[formation_stock_trading_days >= (formation_index_trading_days -10* month_j)].index #暂时month_j=5
        df1_t_square = df1_t_square.loc[df1_t_square['code'].isin(trading_stock_list)]
        # print(df1_t_square)

        # 求t^2前的系数
        t_square_params = thesis_Funtions.ols_param(df1_t_square)
        # 将t^2前的系数加载到一个list中
        t_square_params_list.append(t_square_params)
        # 将股票code放到一个list中
        stock_code_list.append(str(df_temporary['code'][0]))
        # 将形成期累计收益率放到list中
        formation_total_return.append(thesis_Funtions.total_yield(df1_t_square))
        # 将形成期内股票数据append到一起; df_temporary_end是形成期的无t、t²的所有股票数据
        df_temporary_end = df_temporary_end.append(df1_t_square[['code','adjust_open','adjust_price_fr','change','limit_up','traded_market_value']])
    except:
        # print(df_temporary['code'].value_counts().index)
        # print(df1_t_square)
        continue

    # ============================选出持有期阶段对应的股票数据============================
    df_0 = thesis_Funtions.holding_period(df_temporary,initial_date_year=initial_date_year,initial_date_month=initial_date_month,month_j=month_j,month_k=month_k) # df_0是单只持有期股票原始数据
    df_0.reset_index(inplace = True)
    # append所有的持有期股票数据
    df_formation_origin = df_formation_origin.append(df_0) #持有期的所有股票数据，无t^2 无累计收益率
# print(df_temporary_end)
# print(df_formation_origin)
# exit()
# ============================将所有股票的代码和相应的t^2前的系数============================
for i in range(len(stock_code_list)):
    df_temporary_params.loc[i, 'code'] = stock_code_list[i]
    df_temporary_params.loc[i, 't_square'] = t_square_params_list[i]
    df_temporary_params.loc[i,'total_return'] = formation_total_return[i] -1.0  # df_temporary_params是过度性的，
                                                                                # 有code、t^2和形成期的累计收益率，且一直股票只有一行相应数据
    # 将形成期t^2和累计收益率合并入相应股票中
    df_temporary_end = pd.merge(df_formation_origin, df_temporary_params[i:i+1], on='code') # df_temporary_end是持有期单只股票包含t²和形成期的累计收益
    # append起来所有形成期的有t^2和total_return的股票
    df_formation = df_formation.append(df_temporary_end) # 持有期的所有股票+对应形成期的t^2和累计收益率
# print(df_formation.loc[df_formation['code']=='sh600000'])
# exit()


winner_number = int(np.floor(len(df_formation['total_return'].value_counts().index)*0.2)) #第一次分组选出的winner组合的股票数量
winner_number_list=list()
winner_number_list = sorted(df_formation['total_return'].value_counts().index,reverse=True)[:winner_number] #第一次分组选出的winner组合的股票构成的list
df_winner = df_formation[df_formation['total_return'].isin(winner_number_list)] #从持有期股票中选出winner中的股票
# print(df_winner)
# exit()

# ============================对持有期的股票先用total_return进行分组============================
# cat = pd.qcut(df_temporary_params['total_return'],5)
# win_lose_list = list()
# for name,group in df_temporary_params.groupby(by=cat,as_index=False):
#     # name是每个bin的范围，group是对应total_return的值的dataframe(是df_temporary_params的一部分）
#     aa = df_formation.loc[df_formation['total_return'].isin(group['total_return'])] # 在持有期所有股票中，通过对应total_return来选出对应股票持有期数据
#     # print(aa)
#     # exit()
#     win_lose_list.append(aa) # 将分别满足bin范围的股票持有期数据组合放到一个list的不同位置中
#
# # 形成winner和loser组合
# df_winner = win_lose_list[0] #根据形成期累计收益率排序的winner组合
# df_loser = win_lose_list[-1] #根据形成期累计收益率排序的loser组合
# =========================================================================================
#
# 将date设置为index

# 将date设置为index
df_winner.set_index('date',inplace = True)
# print(df_winner)
# exit()
# ============================根据t^2前的系数进行第二次分组============================
accl_num = np.floor(len(df_winner.sort_values(['t_square'],ascending=False)['code'].unique()) * 0.2) #选出前20%的股票数量
winner_stock_name = df_winner.sort_values(['t_square'],ascending=False)['code'].unique() #所有累计收益率排前20%股票的按t^2前系数降序排列后的名称
slice_num = int(accl_num) #用于切片，对钱20%股票数量进行取整
accl_stock_name = winner_stock_name[:slice_num]
df_accl_winner = pd.DataFrame()
for i in accl_stock_name:
    df_accl_winner_temporary = df_winner.loc[df_winner['code'].isin([i])] #在winner组合中选择出t²排名前20%的股票
    if df_accl_winner_temporary['limit_up'][0] == 1: #删除持有期第一个月第一天开盘涨停的那一天，并延后一天持有
        df_accl_winner_temporary = df_accl_winner_temporary.iloc[1:]
    # 得到经过两次分组后的选出股票
    df_accl_winner = df_accl_winner.append(df_accl_winner_temporary)
# print(df_accl_winner)
# exit()

# 求持有期每只股票每日的流通总市值之和
df_traded_market_value_sum = df_accl_winner.groupby('date')['traded_market_value'].sum()
# 修改Series对象的name
df_traded_market_value_sum.rename('traded_market_value_sum',inplace=True)
# print(df_traded_market_value_sum)
# exit()

df_final_total = pd.DataFrame()
# ============================以月份为间隔，聚合============================
for i in df_accl_winner['code'].value_counts().index:
    df_accl_winner_temporary_1 = df_accl_winner.loc[df_accl_winner['code']== i]
    # 将每只股票每日的流通总市值之和添加至每只股票
    df_accl_winner_temporary_1 = df_accl_winner_temporary_1.join(df_traded_market_value_sum,how ='inner')
    # 计算每日的市值加权平均收益率
    df_accl_winner_temporary_1['value_weighted_average_change'] = (df_accl_winner_temporary_1['traded_market_value'] / df_accl_winner_temporary_1['traded_market_value_sum'])\
                                                                  *df_accl_winner_temporary_1['change']
    # 将每只股票以按月聚合
    resampled = df_accl_winner_temporary_1.resample('M')
    # grouped = series_tempory.groupby(lambda x:x.month)

    # 取选定的股票的'code','close','adjust_price_fr','t_square','total_return'每月的第一个交易日的数据
    df_resample = resampled[['code','close','adjust_price_fr','t_square','total_return']].first()
    # 将股票每月的日涨跌幅形成一个list
    df_final = pd.merge(df_resample,pd.DataFrame({'everyday_return':resampled['value_weighted_average_change'].apply(lambda x: np.array(x))})
             ,left_index=True,right_index=True) #持有期ACwinner单只股票数据，有everyday_return

    # ============================求ACwinner组合的等权重持有期总收益率============================
    df_final_total = df_final_total.append(df_final)
# 重新设置index
df_final_total.reset_index(inplace=True)
# 将某股票在某月没交易（无数值）的行删除
df_final_total = df_final_total.loc[~df_final_total['code'].isnull()]
# print(df_final_total)
# exit()

# 创建透视表，columns为组合中每只股票的代码，index为形成期‘XXXX年-XX月’，值为对应每个股票每个月中每日的加权平均收益率的list
df_final_total_return = df_final_total.pivot(index='date',columns='code',values='everyday_return')
# df_final_total_return = df_final_total.pivot(index='date',columns='code',values='everyday_return')
# print(df_final_total_return.columns)
# exit()

# 分别对每只股票进行操作求出每只股票的持有期总收率后等权重加权平均
# weight = 1.0/len(df_final_total_return.columns) #此处若为等权重持有组合中每只股票的情况下设置
ACwinner_group_total_return = 0
for c in range(len(df_final_total_return.columns)):
    test = np.array(df_final_total_return.iloc[:,c]+1.0) # ACwinner中每一只股票的'everyday_return'的数组，对应每个月
    # print(test.shape[0])
    hold_every_momth_total_return = list() # 持有期单只股票的累计总收益率
    print('ACwinner group stocks:', df_final_total_return.columns[c])
    for i in (range(test.shape[0])):
        try:
            every_month_everyday_return = test[i]
            # print(every_month_everyday_return)
            # print(every_month_everyday_return.cumprod())
            # print(every_month_everyday_return.prod())
            # print('*************************************')
            hold_every_momth_total_return.append(every_month_everyday_return.prod()) #持有期单只股票每个月的累计总收益率
        except:
            print('*********************'+'第'+str(i)+'个月无数据'+'***********************')
            # print('ACwinner group stocks:', df_final_total_return.columns[c])
            pass # 防止某只股票在持有期中的个别月份无数据，而出错
    print('ACwinner group stocks:',df_final_total_return.columns[c])
    print('per stock holding total return:',hold_every_momth_total_return)
    weighted_hold_total_return = (np.array(hold_every_momth_total_return).prod()-1)   #  *weight  #持有期单只股票加权平均后的总累计收益率
    print('per stock weighted holding total return:',weighted_hold_total_return)
    ACwinner_group_total_return = ACwinner_group_total_return + weighted_hold_total_return
# 设置结束计时时间
end_time = time.time()
print('====================================================')
print('acwinner_group_total_return_holding_date:',df_final_total_return.index)
print('acwinner_group_total_return:',ACwinner_group_total_return)
print('time of program running:', end_time - start_time)

