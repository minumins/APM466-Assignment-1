import numpy as np
import plotly.graph_objects as go
import bonds_data

jan_6, jan_7, jan_8, jan_9, jan_10 = bonds_data.jan_6_bonds, bonds_data.jan_7_bonds, bonds_data.jan_8_bonds, bonds_data.jan_9_bonds, bonds_data.jan_10_bonds
jan_13, jan_14, jan_15, jan_16, jan_17 = bonds_data.jan_13_bonds, bonds_data.jan_14_bonds, bonds_data.jan_15_bonds, bonds_data.jan_16_bonds, bonds_data.jan_17_bonds

data = [jan_6, jan_7, jan_8, jan_9, jan_10, jan_13, jan_14, jan_15, jan_16, jan_17]

def spot_curve(bond_data):
    spot_rates = np.zeros(len(bond_data))
    for i in range(len(bond_data)):
        price, coupon_rate, maturity, days_since_payment = bond_data[i][0], bond_data[i][1], bond_data[i][2], bond_data[i][3]
        dirty_price = price + (days_since_payment / 365) * coupon_rate 

        # for the first bond maturing in 6 months, calculate the spot rate directly
        if i == 0:
            spot_rates[i] = np.log(dirty_price / (100 + coupon_rate/2)) / ((-1)*maturity)

        # for the rest of the bonds, use previously calculated spot rates
        else:
            for j in range(i):
                t = bond_data[j][2]
                dirty_price -= (coupon_rate/2) * np.exp((-1)*spot_rates[j]*t)
            spot_rates[i] = (-1)*np.log(dirty_price / (100 + coupon_rate/2)) / maturity

    return spot_rates

def yield_curve(bond_data):
    yields = np.zeros(len(bond_data))
    for i in range(len(bond_data)):
        price, coupon_rate, maturity, days_since_payment = bond_data[i][0], bond_data[i][1], bond_data[i][2], bond_data[i][3]
        dirty_price = price + (days_since_payment / 365) * coupon_rate 
        
        if i == 0:
            yields[i] = np.log(dirty_price / (100 + coupon_rate/2)) / ((-1)*maturity)

        else:
            t = 0
            for j in range(i+1):
                t += bond_data[j][2]
            yields[i] = ((-1)*(np.log(dirty_price / (100 + coupon_rate/2)) - i*np.log(coupon_rate/2))) / t

    return yields

def forward_curve(spot_curve):
    forward_rates = np.zeros(7)
    one_year_rate = spot_curve[1]

    for i in range(len(forward_rates)):
        n_index = i + 3
        n = (n_index - 1) / 2
        forward_rates[i] = (spot_curve[n_index] * (n + 1) - one_year_rate) / n

    return forward_rates

# graph for the spot curves
spot_curves = np.empty(len(data), dtype=object)
for i in range(len(data)): 
    spot_curves[i] = spot_curve(data[i])

maturities = [bond[2] for bond in jan_6]

fig = go.Figure()
fig.update_layout(title='Bootstrapped Spot Rate Curve', xaxis_title='Maturity (Years)', yaxis_title='Spot Rate (%)')

for i in range(len(data)):
    if i < 5:
        fig.add_trace(go.Scatter(x=maturities, y=spot_curves[i] * 100, mode='lines+markers', name = 'Jan ' + str(i+6)))
    else: 
        fig.add_trace(go.Scatter(x=maturities, y=spot_curves[i] * 100, mode='lines+markers', name = 'Jan ' + str(i+8)))

fig.show()

# graph for the yield curves
yield_curves = np.empty(len(data), dtype=object)
for i in range(len(data)): 
    yield_curves[i] = yield_curve(data[i])

fig_2 = go.Figure()
fig_2.update_layout(title='Bootstrapped Yield Curve', xaxis_title='Maturity (Years)', yaxis_title='Yield (%)')

for i in range(len(data)):
    if i < 5:
        fig_2.add_trace(go.Scatter(x=maturities, y=yield_curves[i] * 100, mode='lines+markers', name = 'Jan ' + str(i+6)))
    else: 
        fig_2.add_trace(go.Scatter(x=maturities, y=yield_curves[i] * 100, mode='lines+markers', name = 'Jan ' + str(i+8)))

fig_2.show()

# graph for the forward curves
forward_curves = np.empty(len(data), dtype=object)
for i in range(len(data)): 
    forward_curves[i] = forward_curve(spot_curves[i])

forward_years = [1, 1.5, 2, 2.5, 3, 3.5, 4]

fig_3 = go.Figure()
fig_3.update_layout(title='One-Year Forward Curve', xaxis_title='Years', yaxis_title='Forward Rate (%)')

for i in range(len(data)):
    if i < 5:
        fig_3.add_trace(go.Scatter(x=forward_years, y=forward_curves[i] * 100, mode='lines+markers', name = 'Jan ' + str(i+6)))
    else: 
        fig_3.add_trace(go.Scatter(x=forward_years, y=forward_curves[i] * 100, mode='lines+markers', name = 'Jan ' + str(i+8)))

fig_3.show()



# calculating covariance matrices
def yield_log_returns(yield_curves):
    log_returns = np.zeros((5,9))
    for i in range(5):
        for j in range(9):
            log_returns[i][j] = np.log(yield_curves[j + 1][2*i + 1] / yield_curves[j][2*i + 1])
    return log_returns

yield_log_returns = yield_log_returns(yield_curves)
yield_cov_matrix = np.cov(yield_log_returns)
print("Yield Covariance Matrix: ", yield_cov_matrix)

def forward_log_returns(forward_curves):
    log_returns = np.zeros((4,9))
    for i in range(4):
        for j in range(9):
            log_returns[i][j] = np.log(forward_curves[j + 1][2*i] / forward_curves[j][2*i])
    return log_returns

forward_log_returns = forward_log_returns(forward_curves)
forward_cov_matrix = np.cov(forward_log_returns)
print("Forward Covariance Matrix: ", forward_cov_matrix)

# calculating eigenvalues and eigenvectors
yield_eigenvalues, yield_eigenvectors = np.linalg.eig(yield_cov_matrix)
print("Yield Eigenvalues: ", yield_eigenvalues)
print("Yield Eigenvectors: ", yield_eigenvectors)

forward_eigenvalues, forward_eigenvectors = np.linalg.eig(forward_cov_matrix)
print("Forward Eigenvalues: ", forward_eigenvalues)
print("Forward Eigenvectors: ", forward_eigenvectors)