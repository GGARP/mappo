import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import math
import random

class ARMADemand:
    def __init__(self, ar_params, ma_params, n_periods):
        """
        Initialize the ARMA demand model.

        :param ar_params: List of AR coefficients (e.g., [0.5])
        :param ma_params: List of MA coefficients (e.g., [0.3])
        :param n_periods: Number of time periods to generate demand for
        """
        # Define ARMA process
        self.ar_process = ArmaProcess(np.r_[1, -np.array(ar_params)], np.r_[1, np.array(ma_params)])
        self.n_periods = n_periods
    def generate_demand(self):
        """
        Generate ARMA demand series.

        :return: Demand series as a list
        """
        # Generate ARMA component
        arma_series = self.ar_process.generate_sample(nsample=self.n_periods)
        # Ensure non-negative values and round up to the nearest integer
        demand = np.ceil(np.maximum(arma_series, 0)).astype(int)
        return demand

    # Instantiate ARMA demand generator
arma_demand = ARMADemand(
    ar_params=[0.5],
    ma_params=[0.3],
    n_periods=100
)

np.random.seed(0)


def gradual_increase(t, D0=6, r=1):
    """Linearly increasing demand over time."""
    return D0 + r * t

def cyclical_demand(t, A=5, T=12, C=6):
    """Sinusoidal (seasonal) demand fluctuations."""
    return int(round(A * math.sin(2 * math.pi * t / T) + C))

def demand_shock(t, D_base=5, S=8, t0=8, delta_t=2):
    """Demand jumps (or drops) for a short interval starting at t0."""
    if t0 <= t <= t0 + delta_t:
        return D_base + S
    else:
        return D_base 


def simulate_inar1(alpha, lam, n_periods):
    """
    Simulate an INAR(1) process.
    
    The process is defined as:
        X[t] = Binomial(X[t-1], alpha) + ε[t]
    where ε[t] ~ Poisson(lam).

    Parameters:
      alpha (float): Thinning parameter (0 <= alpha <= 1) that determines the survival rate of past counts.
      lam (float): Mean of the Poisson-distributed innovations (new arrivals).
      n_periods (int): Number of time periods to simulate.
      
    Returns:
      numpy.ndarray: An array of integer values representing the INAR(1) process.
    """
    X = np.zeros(n_periods, dtype=int)
    # Initialize the process, e.g., using a Poisson draw for the first period
    X[0] = np.random.poisson(lam)
    
    for t in range(1, n_periods):
        # Thinning: each unit from the previous period survives with probability alpha
        survivors = np.random.binomial(X[t-1], alpha)
        # New arrivals (innovations) are drawn from a Poisson distribution
        new_arrivals = np.random.poisson(lam)
        # The count at time t is the sum of survivors and new arrivals
        X[t] = survivors + new_arrivals
        
    return X

# Set seed for reproducibility
np.random.seed(0)

# Generate the demand series using the INAR(1) process:
INARdemand_series = simulate_inar1(alpha=0.5, lam=2, n_periods=13)
print(INARdemand_series)

env_configs = {
    'two_agent': {
        'num_stages': 2,
        'num_periods': 2,
        'init_inventories': [4, 4],
        'lead_times': [1, 2],
        'demand_fn': lambda t: 4,
        'prod_capacities': [10, 10],
        'sale_prices': [0, 0],
        'order_costs': [0, 0],
        'backlog_costs': [1, 1],
        'holding_costs': [1, 1],
        'stage_names': ['retailer', 'supplier'],
    },
    'constant_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: 4,
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [0, 0, 0, 0],
        'order_costs': [0, 0, 0, 0],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],   
    },
    'variable_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: np.random.randint(0, 5),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [0, 0, 0, 0],
        'order_costs': [0, 0, 0, 0],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],  
    },
    'larger_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [20, 20, 20, 20],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: np.random.randint(0, 9),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'], 
    },
    'seasonal_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: np.random.randint(0, 5) if t <= 4 else np.random.randint(5, 9),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],  
    },
    'normal_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 14, 16, 18],
        'lead_times': [1, 2, 3, 4],
        'demand_fn': lambda t: max(0, int(np.random.normal(4, 2))),
        'prod_capacities': [20, 22, 24, 26],
        'sale_prices': [9, 8, 7, 6],
        'order_costs': [8, 7, 6, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'], 
    },
     'increasing_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [15, 15, 15, 15],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: gradual_increase(t, D0=5, r=1),
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [9, 8, 7, 6],
        'order_costs': [8, 7, 6, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
     'cyclical_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [15, 15, 15, 15],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: cyclical_demand(t, A=5, T=12, C=6),
        'prod_capacities': [20, 25, 30, 35],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
    'demand_shock': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [12, 12, 12, 12],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t: demand_shock(t, D_base=5, S=8, t0=8, delta_t=2),
        'prod_capacities': [20, 22, 24, 26],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },
        'stochastic_demand': {
        'num_stages': 4,
        'num_periods': 12,
        'init_inventories': [20, 20, 20, 20],
        'lead_times': [2, 2, 2, 2],
        'demand_fn': lambda t:  INARdemand_series[t],
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
    },  
}
