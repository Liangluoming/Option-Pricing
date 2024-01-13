"""
@author: Liang Luoming
@email: liangluoming00@163.com
"""

import numpy as np 
import pandas as pd 
from scipy.stats import norm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

def bsm_pricing(S, K, T, r, vol, q = 0.0, option_type = "call"):
    """
    pricing based on blackscholes model
    params:
        S: 标的资产价格, K: 执行价格, T: 到期时间(需经过年化处理, 最简单的年化处理是日度/252), r: 无风险利率 vol: 隐含波动率 q: 标的资产收益率
        option_type = "call" or "put"
    return:
        期权价格
    """
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "call" : 
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        return ValueError
    return price

def bsm_vega(S, K, T, r, vol):
    """
    基于BSM的Vega, 即标的资产价格对于波动率的一阶导数
    """
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def get_impliedVol(market_price, S, K, T, r, initial_vol = 0.3, q = 0.0, option_type = "call", max_iter = 1000, epsilon = 1e-4):
    """
    牛顿迭代法求隐含波动率（一般的二分法也可以）
    params:
        market_prices: 观察到的市场上期权价格
        S: 标的资产价格, K: 执行价格, T: 到期时间(需经过年化处理, 最简单的年化处理是日度/252), r: 无风险利率 vol: 隐含波动率 q: 标的资产收益率
        initial_vol: 设定的初始波动率, 默认值为30%
        option_type = "call" or "put"
        max_iter: 最大迭代次数, 默认值为1000
        epsilon: 收敛精度, 默认值为0.0001
    """
    for _ in range(max_iter):
        price = bsm_pricing(S, K, T, r, initial_vol, q, option_type)
        vega = bsm_vega(S, K, T, r, initial_vol)
        error = market_price - price
        if abs(error) < epsilon:
            return initial_vol
        initial_vol += error / vega
    return initial_vol

def impliedVol_curve(market_prices, K_series, S, T, r, initial_vol = 0.3, q = 0.0, option_type = "call", max_iter = 1000, epsilon = 1e-4):
    """
    隐含波动率曲线
    params:
        market_prices: 观察到的市场上期权价格序列
        K_series: 观察到的市场上期权执行价格序列
        S: 标的资产价格, K: 执行价格, T: 到期时间(需经过年化处理, 最简单的年化处理是日度/252), r: 无风险利率 vol: 隐含波动率 q: 标的资产收益率
        initial_vol: 设定的初始波动率, 默认值为30%
        option_type = "call" or "put"
        max_iter: 最大迭代次数, 默认值为1000
        epsilon: 收敛精度, 默认值为0.0001

    其实这里可以进行进一步优化: 利用numpy的向量化运算简化for循环(有空再调整)
    """
    impliedVols = []
    for mp, k in zip(market_prices, K_series):
        impliedVol = get_impliedVol(mp, S, k, T, r, initial_vol, q, option_type, max_iter, epsilon)
        impliedVols.append(impliedVol)
    return impliedVols

def impliedVar_Interpolation(df_surface, T):
    """
    隐含方差线性插值
    Params:
        df_surface: 一个索引为到期时间, 列名为执行价格的隐含波动率查询表(需经过年化处理, 最简单的年化处理是日度/252)
        T: 为到期时间(需经过年化处理, 最简单的年化处理是日度/252)
    """
    if len(df_surface[df_surface.index < T]) and len(df_surface[df_surface.index > T]):
        vol_T1 = df_surface[df_surface.index < T].iloc[-1, :]
        vol_T2 = df_surface[df_surface.index > T].iloc[0, :]
        var = vol_T1.values ** 2 + (T - vol_T1.index) / (vol_T2.index - vol_T1.index) * (vol_T2.values ** 2 - vol_T1.values ** 2)
        vol_T = np.sqrt(var)
        return vol_T
    else:
        return ValueError

def CubicSpline_impliedVol_surface(market_prices, K_series, Ts, S, r, initial_vol = 0.3, q = 0.0, option_type = "call", max_iter = 1000, epsilon = 1e-4, K_gap = 10, T_gap = 5):
    """
    三次样条函数构建隐含波动率曲面
    params:
        market_prices: 观察到的市场上期权价格序列
        K_series: 观察到的市场上期权执行价格序列
        Ts: 到期时间序列(需经过年化处理, 最简单的年化处理是日度/252)
        S: 标的资产价格, r: 无风险利率 vol: 隐含波动率 q: 标的资产收益率
        initial_vol: 设定的初始波动率, 默认值为30%
        option_type = "call" or "put"
        max_iter: 最大迭代次数, 默认值为1000
        epsilon: 收敛精度, 默认值为0.0001
        K_gap: 把观察到的执行价格区间按K_gap切分, 默认为10
        T_gap: 把观察到的到期时间序列按T_gap切分, 默认为5
    """
    K_interval = np.arange(min(K_series),max(K_series) + K_gap, K_gap)
    K = np.array(set(K_series).union(set(K_interval)))
    df_surface = pd.DataFrame()
    for t in Ts:

        impliedVols = impliedVol_curve(market_prices, K_series, S, t, r, initial_vol, q, option_type, max_iter, epsilon)
        cs = CubicSpline(K_series, impliedVols)
        impliedVols = cs(K)
        df = pd.DataFrame(data = impliedVols, index = t, column = K)
        df_surface = pd.concat([df_surface, df])
    
    T_interval = np.arange(min(Ts), max(Ts) + T_gap, T_gap)
    vol_T_interval = []
    for ti in T_interval:
        vol_ti = impliedVar_Interpolation(df_surface, ti)
        vol_T_interval.append(vol_ti)
    
    df_surface = pd.concat([df_surface, vol_T_interval])

    fig = plt.figure()
    x, y = np.meshgrid(K_interval, Ts)
    z = df_surface.values
    ax = fig.add_subplot(111, projection = '3d')
    surface = ax.plot_surface(x, y, z, cmap = 'viridis')
    fig.colorbar(surface)
    ax.set_xlabel('K')
    ax.set_ylabel('T')
    ax.set_zlabel('implied_vol')
    plt.show()

    return df_surface
    
