import numpy as np
import math

# Create a dedicated random number generator with a fixed seed.
rng = np.random.default_rng(0)

def gradual_increase(t, D0=5, r=1):
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

def simulate_inar1(alpha, lam, n_periods, rng):
    """
    Simulate an INAR(1) process.
    
    The process is defined as:
        X[t] = Binomial(X[t-1], alpha) + ε[t]
    where ε[t] ~ Poisson(lam).
    """
    X = np.zeros(n_periods, dtype=int)
    # Initialize with a Poisson draw for the first period.
    X[0] = rng.poisson(lam)
    
    for t in range(1, n_periods):
        survivors = rng.binomial(X[t-1], alpha)
        new_arrivals = rng.poisson(lam)
        X[t] = survivors + new_arrivals
    return X

# Generate the INAR(1) demand series using the dedicated RNG:
INARdemand_series = simulate_inar1(alpha=0.5, lam=2, n_periods=13, rng=rng)
print("INAR demand series:", INARdemand_series)

# Use the same RNG in all demand functions for replicability.
env_configs = {
    'two_agent': {
        'num_stages': 2,
        'num_periods': 2,
        'init_inventories': [4, 4],
        'lead_times': [1, 2],
        'demand_fn': lambda t: 4,  # constant demand
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
        # Use the dedicated RNG here instead of np.random
        'demand_fn': lambda t: rng.integers(0, 5),
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
        # Use the dedicated RNG here as well
        'demand_fn': lambda t: rng.integers(0, 9),
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
        # Conditional RNG calls using the same generator
        'demand_fn': lambda t: rng.integers(0, 5) if t <= 4 else rng.integers(5, 9),
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
        # Using the dedicated RNG for normal distribution
        'demand_fn': lambda t: max(0, int(rng.normal(4, 2))),
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
        # This function is deterministic
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
        # Here we use the pre-generated INARdemand_series.
        'demand_fn': lambda t: INARdemand_series[t],
        'prod_capacities': [20, 20, 20, 20],
        'sale_prices': [5, 5, 5, 5],
        'order_costs': [5, 5, 5, 5],
        'backlog_costs': [1, 1, 1, 1],
        'holding_costs': [1, 1, 1, 1],
        'stage_names': ['retailer', 'wholesaler', 'distributor', 'manufacturer'],  
    },        
}

# Example: Evaluate one of the demand functions.
# This will now be reproducible thanks to the dedicated RNG.
print("Variable demand for t=0:", env_configs['variable_demand']['demand_fn'](0))
