import numpy as np

# use euler scheme here, maybe should use Milstein
def simulate_rough_heston(S0, T, params, r=0.045, n_steps=200, n_sims=10000):
    """
    Monte Carlo simulation for the Rough Heston model.
    params = (alpha, lambd, rho, nu, theta, V0)
    """
    alpha, lambd, rho, nu, theta, V0 = params
    H = alpha - 0.5
    dt = T / n_steps
    
    # 1. Correlated Brownian Motions
    Z1 = np.random.normal(0, 1, (n_sims, n_steps))
    Z2 = np.random.normal(0, 1, (n_sims, n_steps))
    W = Z1 * np.sqrt(dt)
    B = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
    
    S = np.full(n_sims, S0, dtype=float)
    V = np.full((n_sims, n_steps + 1), V0, dtype=float)
    
    # 2. Kernel-based Variance Simulation (Euler approximation)
    # Note: For H < 0.5, the kernel (t-s)^{H-0.5} is singular at s=t.
    for i in range(n_steps):
        # Update Spot
        S *= np.exp((r - 0.5 * V[:, i]) * dt + np.sqrt(V[:, i]) * W[:, i])
        
        # Update Variance using the integral form from the 2nd paper (Eq 3.1)
        # We approximate the integral with a Riemann sum
        curr_t = (i + 1) * dt
        times = np.linspace(0, curr_t, i + 1)
        kernel = (curr_t - times)**(H - 0.5) / 1.0 # Gamma factor simplified for speed
        
        # Stochastic integral part
        # V_t = V0 + Int K(t-s) [lambda(theta - V)ds + nu sqrt(V)dB]
        drift = lambd * (theta - V[:, :i+1])
        diffusion = nu * np.sqrt(np.maximum(V[:, :i+1], 0))
        
        # Simple weighted sum over history
        # (This is a naive Euler; hybrid schemes are more accurate but complex)
        int_term = np.sum(kernel[:-1] * (drift[:, :-1] * dt + diffusion[:, :-1] * B[:, :i]), axis=1)
        V[:, i+1] = np.maximum(V0 + int_term, 1e-6)
        
    return S

def price_call_mc(S0, K, T, r, params, n_sims=10000):
    S_final = simulate_rough_heston(S0, T, params, r=r, n_sims=n_sims)
    payoffs = np.maximum(S_final - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def price_put_mc(S0, K, T, r, params, n_sims=10000):
    S_final = simulate_rough_heston(S0, T, params, r=r, n_sims=n_sims)
    payoffs = np.maximum(K - S_final, 0)
    return np.exp(-r * T) * np.mean(payoffs)
