import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma
from numpy.random import default_rng
from scipy.spatial import distance_matrix

# Set random seed for reproducibility
rng = default_rng(seed=42)

"Using Gaussian Random Field r(x) for locations x on the grid of size 50 × 50 with mean 0 and variance 1,"
"I am going to use the powered exponential and Matern correlation functions rho_r(tau)"
"to generate and show realizations of trunctated GRFs. "

# ----- Simulating Gaussian Random Fields (GRFs) -----
# Define GRF Parameters
mu_r = 0            # Mean of GRF
sigma2_r = 1        # Variance of GRF
grid_size = 50      # Grid dimensions (50x50)
x = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, x) # Shape (50, 50)
coords = np.column_stack([X.ravel(), Y.ravel()])  # Shape (2500, 2)

#Compute pairwise distances
tau = cdist(coords, coords) / 0.1  # tau = ||x - x'|| / 0.1 (Shape (2500, 2500))
tau[tau == 0] = 1e-10  # avoid division issues in Matérn

# Define correlation functions
def powered_exponential(tau, nu):
    """
    Powered Exponential correlation function.
    """
    return np.exp(-tau ** nu)

def matern(tau, nu):
    """
    Matérn correlation function.
    """
    corr = ((2 ** (1 - nu)) / gamma(nu)) * ((np.sqrt(2 * nu) * tau) ** nu) * kv(nu, (np.sqrt(2 * nu) * tau))
    corr[np.isnan(corr)] = 1  # handle zero distance manually
    np.fill_diagonal(corr, 1)  # ensure diagonal = 1
    return corr

# Simulate one GRF realization
def simulate_grf(corr_func, tau, nu, sigma2, mu=0):
    """
    Simulates a single realization of a Gaussian Random Field.
    """
    corr_matrix = corr_func(tau, nu) # Correlation matrix (shape (2500, 2500))
    cov_matrix = sigma2 * corr_matrix # Covariance matrix 
    sample = rng.multivariate_normal(mean=np.full(tau.shape[0], mu), cov=cov_matrix) # Draw random samples from a multivariate normal dist.
    return sample.reshape((grid_size, grid_size)) # Reshape to (50, 50) and return sample of GRF

def plot_grf(field, title):
    """
    Plots a simulated GRF realization using matplotlib.
    """
    plt.figure()
    plt.imshow(field, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    plt.colorbar(label='r(x)')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(title, dpi=300, bbox_inches='tight')

# -- Run simulations and plot --

# Powered Exponential ν = 1
grf_pe_1 = simulate_grf(powered_exponential, tau, nu=1, sigma2=sigma2_r)
plot_grf(grf_pe_1, 'Powered Exponential (ν=1)')

# Powered Exponential ν = 1.9
grf_pe_1_9 = simulate_grf(powered_exponential, tau, nu=1.9, sigma2=sigma2_r)
plot_grf(grf_pe_1_9, 'Powered Exponential (ν=1,9)')

# Matérn ν = 1
grf_matern_1 = simulate_grf(matern, tau, nu=1, sigma2=sigma2_r)
plot_grf(grf_matern_1, 'Matérn (ν=1)')

# Matérn ν = 3
grf_matern_3 = simulate_grf(matern, tau, nu=3, sigma2=sigma2_r)
plot_grf(grf_matern_3, 'Matérn (ν=3)')

# ----- Truncate GRF into TGRF -----

def truncate_grf(grf_matrix):
    """
    Truncates the GRF: 1 if r(x) <= 0, else 0.
    """
    return (grf_matrix <= 0).astype(int)

def plot_tgrf(tgrf_matrix, title):
    """
    Plots a binary TGRF field using black/white colormap.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(tgrf_matrix, origin='lower', cmap='gray_r', extent=[0, 1, 0, 1])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(ticks=[0, 1], label='l(x)')
    plt.clim(-0.5, 1.5)
    plt.tight_layout()
    plt.savefig(title, dpi=300, bbox_inches='tight')

# Truncate each GRF to obtain the corresponding TGRF (Apply truncation and plot)
tgrf_pe_1 = truncate_grf(grf_pe_1)
plot_tgrf(tgrf_pe_1, 'TGRF: Powered Exponential (ν = 1)')

tgrf_pe_1_9 = truncate_grf(grf_pe_1_9)
plot_tgrf(tgrf_pe_1_9, 'TGRF: Powered Exponential (ν = 1,9)')

tgrf_matern_1 = truncate_grf(grf_matern_1)
plot_tgrf(tgrf_matern_1, 'TGRF: Matérn (ν = 1)')

tgrf_matern_3 = truncate_grf(grf_matern_3)
plot_tgrf(tgrf_matern_3, 'TGRF: Matérn (ν = 3)')

# ----- Compute empirical correlations for all TGRFs -----

grid_size_new = 30 # Smaller grid for faster computation
max_lag = 25
lag_step = 1

# Recalculate coordinates and distance matrix
x_new = np.arange(grid_size_new)
y_new = np.arange(grid_size_new)
xx_new, yy_new = np.meshgrid(x_new, y_new)                     # Shape (30, 30)
coords_new = np.column_stack([xx_new.ravel(), yy_new.ravel()]) # Shape (900, 2)
dists_new = distance_matrix(coords_new, coords_new)            # Shape (900, 900)

# Downsample the TGRFs from 50x50 to 30x30
def downsample_grf(field, new_size):
    """
    Downsamples a 2D field to (new_size, new_size) using slicing.
    Assumes field is square and new_size < field.shape[0]
    """
    original_size = field.shape[0]
    #factor = original_size / new_size
    indices = (np.linspace(0, original_size - 1, new_size)).astype(int) # indices for downsampling 
    return field[np.ix_(indices, indices)] # Select rows and columns based on indices (np.ix_ provides a meshgrid)

# Apply downsampling to each TGRF
tgrf_pe_1_new = downsample_grf(tgrf_pe_1, grid_size_new)
tgrf_pe_1_9_new = downsample_grf(tgrf_pe_1_9, grid_size_new)
tgrf_matern_1_new = downsample_grf(tgrf_matern_1, grid_size_new)
tgrf_matern_3_new = downsample_grf(tgrf_matern_3, grid_size_new)

# Comute empirical spatial correlation 
def empirical_correlation(field_matrix, dists=dists_new, max_lag=25, lag_step=1):
    """
    Computes empirical spatial correlation of a 2D binary field.
    """
    values = field_matrix.flatten()  # Truncated GRF values (0 or 1) flattened to 1D
    values_centered = values - np.mean(values)
    product_matrix = np.outer(values_centered, values_centered) # Matrix of pairwise outer product 
    
    lags = np.arange(0, max_lag + lag_step, lag_step) # Lags from 0 to max_lag 
    correlations = [] # List to store correlations for each lag

    for lag_val in lags:
        # Create mask for distances within the lag range
        mask = (dists >= lag_val) & (dists < lag_val + lag_step)
        if np.any(mask): # Check if there are any valid distances
            # Compute empirical correlation (E[r(x)r(x')] / Var[r(x)])
            corr = np.mean(product_matrix[mask]) / np.var(values)
            correlations.append(corr)
        else: # If no valid distances, append NaN
            correlations.append(np.nan)
    
    return lags, correlations

def plot_empirical_corr(lags, corr, title):
    plt.figure()
    plt.plot(lags, corr, marker='o', color='blue')
    plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Empirical Correlation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title, dpi=300, bbox_inches='tight')

#  Compute and plot correlations for all TGRFs
lags1, corr1 = empirical_correlation(tgrf_pe_1_new, dists_new, max_lag, lag_step)
plot_empirical_corr(lags1, corr1, "Empirical Correlation (TGRF Powered Exp ν = 1)")

lags2, corr2 = empirical_correlation(tgrf_pe_1_9_new, dists_new, max_lag, lag_step)
plot_empirical_corr(lags2, corr2, "Empirical Correlation (TGRF Powered Exp ν = 1,9)")

lags3, corr3 = empirical_correlation(tgrf_matern_1_new, dists_new, max_lag, lag_step)
plot_empirical_corr(lags3, corr3, "Empirical Correlation (TGRF Matérn ν = 1)")

lags4, corr4 = empirical_correlation(tgrf_matern_3_new, dists_new, max_lag, lag_step)
plot_empirical_corr(lags4, corr4, "Empirical Correlation (TGRF Matérn ν = 3)")

# ----- End of Code -----