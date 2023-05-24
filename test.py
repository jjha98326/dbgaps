import numpy as np

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

# # 결과 출력
# for asset, weight in zip(assets, portfolio_weights):
#     print(f"{asset}: {weight:.2f}")

print(portfolio_weights)
print(sum(portfolio_weights))