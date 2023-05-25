from time import time
import numpy as np
import pandas as pd
import math
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt
from IPython.display import display


plt.style.use('seaborn')
mpl.rcParams['font.family']='serif'
plt.show()

def gen_path(S0, r, sigma, T, M, I):
    '''기하 브라운 운동에 대한 몬테카를로 경로를 생성
    인수
    =======
    S0: float
        초기 주가/지수 수준
    r: float
        고정 단기 이자율
    sigma: float
        고정 변동성
    T: float
        최종 시간
    M: int
        시간구간의 개수
    I: int
        시뮬레이션 경로의 개수
    반환값
    =======
    paths: ndarray, shape(M+1,I)
        주어진 인수로 시뮬레이션한 경로
    '''
    dt= T/M
    paths=np.zeros((M+1,I))
    paths[0]=S0
    for t in range(1, M+1):
        rand=np.random.standard_normal(I)
        rand=(rand-rand.mean())/rand.std()
        paths[t]=paths[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*rand)
    return paths

def normality_tests(arr):
    '''주어진 데이터 분포의 정규성 검정
    인수
    ======
    array: ndarray
    통계를 생성할 대상 객체
    '''
    print('Skew of data set %14.3f' % scs.skew(arr))
    print('Skew test p-value %14.3f' % scs.skewtest(arr)[1])
    print('Kurt of data set %14.3f' % scs.kurtosis(arr))
    print('Kurt test p-value %14.3f' % scs.kurtosistest(arr)[1])
    print('Norm test p-value %14.3f' % scs.normaltest(arr)[1])

def print_statistics(array):
    '''선택한 통계를 출력
    인수
    ======
    array: ndarray
    통계를 생성할 대상 객체
    '''
    sta=scs.describe(array)
    print('%14s %15s' % ('statistic','value'))
    print(30*'-')
    print('%14s %15.5f' % ('size',sta[0]))
    print('%14s %15.5f' % ('min',sta[1][0]))
    print('%14s %15.5f' % ('max',sta[1][1]))
    print('%14s %15.5f' % ('mean',sta[2]))
    print('%14s %15.5f' % ('std',np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew',sta[4]))
    print('%14s %15.5f' % ('kurtosis',sta[5]))
    
raw=pd.read_csv('C:\\Users\\jjha9\\desktop\\dbgaps.csv', encoding='cp949', index_col=0, parse_dates=True).dropna()
symbols=['KODEX 200','TIGER 코스닥150','TIGER 미국S&P500선물(H)','TIGER 유로스탁스50(합성 H)', 'ACE 일본Nikkei225(H)','TIGER 차이나CSI300','KOSEF 국고채10년','KBSTAR 중기우량회사채','TIGER 단기선진하이일드(합성 H)','KODEX 골드선물(H)','TIGER 원유선물Enhanced(H)','KODEX 인버스','KOSEF 미국달러선물','KOSEF 미국달러선물인버스','KOSEF 단기자금']

data=raw[symbols]
data=data.astype('float')
data=data.dropna()
data.info()
(data/data.iloc[0]*100).plot(figsize=(10,6))
# plt.show()

log_returns=np.log(data/data.shift(1))
log_returns.head()
log_returns.hist(bins=50, figsize=(10,8))
# plt.show()

for sym in symbols:
    print('\nResults for symbol {}'.format(sym))
    print(30*'-')
    log_data=np.array(log_returns[sym].dropna())
    print_statistics(log_data)

sm.qqplot(log_returns['KODEX 200'].dropna(),line='s')
plt.title('KODEX 200')
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles');
# plt.show()

for sym in symbols:
    print('\nResults for symbol {}'.format(sym))
    print(32*'-')
    log_data=np.array(log_returns[sym].dropna())
    normality_tests(log_data)

noa=len(symbols)

def adjust_portfolio_weights():
    # 자산 그룹과 제약 조건 정의
    asset_groups = {
        '국내증권': {'range': (0.1, 0.4), 'count': 2},
        '해외증권': {'range': (0.1, 0.4), 'count': 4},
        '채권': {'range': (0.2, 0.6), 'count': 3},
        '현물자산': {'range': (0.05, 0.2), 'count': 2},
        '국내인버스': {'range': (0, 0.2), 'count': 1},
        '달러': {'range': (0, 0.2), 'count': 2},
        '현금': {'range': (0.01, 0.5), 'count': 1}
    }

    assets = []
    bounds = []

    # 자산 그룹별로 자산 및 제약 조건 추가
    for group, params in asset_groups.items():
        for i in range(params['count']):
            asset_name = f'자산{i + 1}_{group}'
            assets.append(asset_name)
            bounds.append(params['range'])

    # 비중 조정을 위한 랜덤한 비중 생성
    weights = np.random.uniform(low=0, high=1, size=len(assets))

    # 비중 정규화 및 조건 벗어난 비중 조정
    weights /= np.sum(weights)  # 정규화

    for i, (low, high) in enumerate(bounds):
        if weights[i] < low:
            weights[i] = low
        elif weights[i] > high:
            weights[i] = high

    # 합이 1이 되도록 조정
    weights /= np.sum(weights)

    return weights

# 포트폴리오 비중 조정
portfolio_weights = adjust_portfolio_weights()

print(portfolio_weights)
print(sum(portfolio_weights))

# weights=np.random.random(noa)
# weights/=np.sum(weights)
np.dot(portfolio_weights.T,np.dot(log_returns.cov()*124,portfolio_weights))

def port_ret(portfolio_weights):
    return np.sum(log_returns.mean()*portfolio_weights)*124

def port_vol(portfolio_weights):
    return np.sqrt(np.dot(portfolio_weights.T,np.dot(log_returns.cov()*124,portfolio_weights)))

prets=[]
pvols=[]
for p in range(10000):
    def adjust_portfolio_weights():
    # 자산 그룹과 제약 조건 정의
        asset_groups = {
            '국내증권': {'range': (0.1, 0.4), 'count': 2},
            '해외증권': {'range': (0.1, 0.4), 'count': 4},
            '채권': {'range': (0.2, 0.6), 'count': 3},
            '현물자산': {'range': (0.05, 0.2), 'count': 2},
            '국내인버스': {'range': (0, 0.2), 'count': 1},
            '달러': {'range': (0, 0.2), 'count': 2},
            '현금': {'range': (0.01, 0.5), 'count': 1}
        }

        assets = []
        bounds = []

    # 자산 그룹별로 자산 및 제약 조건 추가
        for group, params in asset_groups.items():
            for i in range(params['count']):
                asset_name = f'자산{i + 1}_{group}'
                assets.append(asset_name)
                bounds.append(params['range'])

    # 비중 조정을 위한 랜덤한 비중 생성
        weights = np.random.uniform(low=0, high=1, size=len(assets))

    # 비중 정규화 및 조건 벗어난 비중 조정
        weights /= np.sum(weights)  # 정규화

        for i, (low, high) in enumerate(bounds):
            if weights[i] < low:
                weights[i] = low
            elif weights[i] > high:
                weights[i] = high

    # 합이 1이 되도록 조정
        weights /= np.sum(weights)

        return weights

# 포트폴리오 비중 조정
    portfolio_weights = adjust_portfolio_weights()

    prets.append(port_ret(portfolio_weights))
    pvols.append(port_vol(portfolio_weights))
prets=np.array(prets)
pvols=np.array(pvols)
prets_minus_rf=prets-0.03455
plt.figure(figsize=(10,6))
plt.scatter(pvols,prets, c=prets_minus_rf/pvols, marker='o', cmap='coolwarm')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

a=port_ret(portfolio_weights)-0.03455

import scipy.optimize as sco
def min_func_sharp(portfolio_weights):
    return -a/port_vol(portfolio_weights)

# 자산 그룹별 비중 제약 조건 설정 (예시)
constraints = [
    {'type': 'ineq', 'fun': lambda x: 0.1 - x[0] - x[1]},  # 국내증권 그룹
    {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 0.4},
    {'type': 'ineq', 'fun': lambda x: 0.1 - x[2] - x[3] - x[4] - x[5]},  # 해외증권 그룹
    {'type': 'ineq', 'fun': lambda x: x[2] + x[3] + x[4] + x[5] - 0.4},
    {'type': 'ineq', 'fun': lambda x: 0.2 - x[6] - x[7] - x[8]},  # 채권 그룹
    {'type': 'ineq', 'fun': lambda x: x[6] + x[7] + x[8] - 0.6},
    {'type': 'ineq', 'fun': lambda x: 0.05 - x[9] - x[10]},  # 현물자산 그룹
    {'type': 'ineq', 'fun': lambda x: x[9] + x[10] - 0.2},
    {'type': 'ineq', 'fun': lambda x: x[11] - 0.2},  # 국내인버스 그룹
    {'type': 'ineq', 'fun': lambda x: x[12] +x[13] - 0.2},  # 달러 그룹
    {'type': 'ineq', 'fun': lambda x: 0.01 - x[14]},  # 현금 그룹
    {'type': 'ineq', 'fun': lambda x: x[14] - 0.5},
    {"type": "eq", "fun": lambda x: x.sum() - 1}
    ]

bnds=tuple((0,1) for x in range(noa))
initial_weights = np.ones(15) / 15  # 모든 자산에 대한 초기 비중


print(min_func_sharp(portfolio_weights))

opts=sco.minimize(min_func_sharp, initial_weights, method='SLSQP', bounds=bnds, constraints=constraints)
print(port_ret(opts['x']).round(3))
print(port_vol(opts['x']).round(3))
print(port_ret(opts['x'])/port_vol(opts['x']))
weight_result=opts.x
for i in range(len(weight_result)):
    print(f'자산{i+1}:{weight_result[i]}')