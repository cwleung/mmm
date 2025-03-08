import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def generate_synthetic_data(seed=42, n_weeks=104):
    """
    Generate synthetic marketing and sales data.

    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    n_weeks : int
        Number of weeks to simulate

    Returns:
    --------
    pd.DataFrame, dict
        DataFrame with simulated data and dictionary with true parameter values
    """
    np.random.seed(seed)
    time_idx = np.arange(n_weeks)

    # Simulate weekly media spends
    u_tv = np.random.rand(n_weeks)
    tv_spend = np.where(u_tv > 0.7, u_tv * 300, u_tv * 150)         # TV has frequent high spends

    u_dig = np.random.rand(n_weeks)
    digital_spend = np.where(u_dig > 0.5, u_dig * 80, u_dig * 40)   # Digital always on with moderate variance

    u_print = np.random.rand(n_weeks)
    print_spend = np.where(u_print > 0.85, u_print * 60, u_print * 10)  # Print mostly low, occasionally high

    # True adstock decay parameters for carryover effect
    alpha_tv, alpha_dig, alpha_print = 0.7, 0.5, 0.3

    # Compute adstocked media (carryover effect)
    adstock_tv = apply_adstock(tv_spend, alpha_tv)
    adstock_dig = apply_adstock(digital_spend, alpha_dig)
    adstock_print = apply_adstock(print_spend, alpha_print)

    # Seasonality (annual periodicity) and trend
    seasonal = 15 * np.sin(2*np.pi*time_idx/52) + 10 * np.cos(2*np.pi*time_idx/52)
    trend = 0.2 * time_idx  # linear trend (gradual growth in baseline sales)

    # True coefficients for media effect on sales
    beta_tv, beta_dig, beta_print = 0.08, 0.05, 0.02
    intercept = 80  # baseline sales

    # Simulate sales as sum of baseline, trend, seasonality, media contributions, and noise
    sales = (intercept + trend + seasonal
             + beta_tv * adstock_tv
             + beta_dig * adstock_dig
             + beta_print * adstock_print
             + np.random.normal(0, 5, size=n_weeks))  # Gaussian noise

    # Combine into a DataFrame
    data = pd.DataFrame({
        'week': time_idx,
        'tv_spend': tv_spend,
        'digital_spend': digital_spend,
        'print_spend': print_spend,
        'sales': sales,
        'adstock_tv': adstock_tv,
        'adstock_dig': adstock_dig,
        'adstock_print': adstock_print
    })

    # Store true parameters for later comparison
    true_params = {
        'intercept': intercept,
        'beta_trend': 0.2,
        'beta_sin': 15,
        'beta_cos': 10,
        'beta_tv': beta_tv,
        'beta_dig': beta_dig,
        'beta_print': beta_print,
        'alpha_tv': alpha_tv,
        'alpha_dig': alpha_dig,
        'alpha_print': alpha_print
    }

    return data, true_params

def apply_adstock(media_spend, decay_rate):
    """
    Apply adstock transformation to media spend using geometric decay.

    Parameters:
    -----------
    media_spend : array-like
        Original media spend values
    decay_rate : float
        Decay parameter (between 0 and 1)

    Returns:
    --------
    np.array
        Adstocked media values
    """
    n = len(media_spend)
    adstocked = np.zeros(n)
    adstocked[0] = media_spend[0]

    for t in range(1, n):
        adstocked[t] = media_spend[t] + decay_rate * adstocked[t-1]

    return adstocked

def create_mmm_model(data):
    """
    Create PyMC Marketing Mix Model with adstocked media variables.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing sales and media data

    Returns:
    --------
    pymc.Model
        PyMC model for MMM analysis
    """
    # Prepare regressors for seasonality and trend
    sin52 = np.sin(2*np.pi*data['week']/52)
    cos52 = np.cos(2*np.pi*data['week']/52)

    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', 0, 100)          # baseline intercept
        beta_trend = pm.Normal('beta_trend', 0, 1)          # trend coefficient
        beta_sin = pm.Normal('beta_sin', 0, 10)             # seasonal sine coeff
        beta_cos = pm.Normal('beta_cos', 0, 10)             # seasonal cosine coeff
        beta_tv = pm.Normal('beta_tv', 0, 0.1)              # TV effect coeff
        beta_dig = pm.Normal('beta_dig', 0, 0.1)            # Digital effect coeff
        beta_print = pm.Normal('beta_print', 0, 0.1)        # Print effect coeff
        sigma = pm.HalfNormal('sigma', 10)                  # noise std dev

        # Expected mean sales incorporating all effects
        mu = (intercept
              + beta_trend * data['week'].values
              + beta_sin * sin52
              + beta_cos * cos52
              + beta_tv * data['adstock_tv'].values
              + beta_dig * data['adstock_dig'].values
              + beta_print * data['adstock_print'].values)

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data['sales'])

    return model

def fit_model(model, seed=42, chains=2, tune=1000, draws=2000, target_accept=0.9):
    """
    Fit the PyMC model using MCMC sampling.

    Parameters:
    -----------
    model : pymc.Model
        PyMC model to fit
    seed : int or list
        Random seed(s) for sampling
    chains : int
        Number of MCMC chains
    tune : int
        Number of tuning steps
    draws : int
        Number of samples to draw
    target_accept : float
        Target acceptance rate

    Returns:
    --------
    InferenceData
        ArviZ InferenceData object containing posterior samples
    """
    with model:
        # Handle different seed formats
        if isinstance(seed, int):
            seeds = [seed + i for i in range(chains)]
        else:
            seeds = seed

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=chains,
            random_seed=seeds
        )

    return trace

def compute_contributions(data, trace):
    """
    Compute media contributions using posterior estimates.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with sales and media data
    trace : InferenceData
        ArviZ InferenceData with posterior samples

    Returns:
    --------
    dict
        Dictionary containing contribution time series and summary metrics
    """
    # Calculate seasonal components
    sin52 = np.sin(2*np.pi*data['week']/52)
    cos52 = np.cos(2*np.pi*data['week']/52)

    # Extract posterior means
    beta_tv_mean = trace.posterior['beta_tv'].mean().item()
    beta_dig_mean = trace.posterior['beta_dig'].mean().item()
    beta_print_mean = trace.posterior['beta_print'].mean().item()
    intercept_mean = trace.posterior['intercept'].mean().item()
    beta_trend_mean = trace.posterior['beta_trend'].mean().item()
    beta_sin_mean = trace.posterior['beta_sin'].mean().item()
    beta_cos_mean = trace.posterior['beta_cos'].mean().item()

    # Compute contributions
    tv_contrib = beta_tv_mean * data['adstock_tv']
    dig_contrib = beta_dig_mean * data['adstock_dig']
    print_contrib = beta_print_mean * data['adstock_print']
    base_contrib = (
            intercept_mean
            + beta_trend_mean * data['week']
            + beta_sin_mean * sin52
            + beta_cos_mean * cos52
    )

    contributions = {
        'base': base_contrib,
        'tv': tv_contrib,
        'digital': dig_contrib,
        'print': print_contrib,
        'summary': {
            'total_tv': tv_contrib.sum(),
            'total_digital': dig_contrib.sum(),
            'total_print': print_contrib.sum()
        }
    }

    return contributions

def plot_contributions(data, contributions):
    """
    Create a stacked area plot showing decomposition of sales.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with weeks and sales data
    contributions : dict
        Dictionary with contribution time series

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    weeks = data['week']
    plt.stackplot(
        weeks,
        contributions['base'],
        contributions['tv'],
        contributions['digital'],
        contributions['print'],
        labels=['Base/Seasonality', 'TV', 'Digital', 'Print'],
        colors=['#cccccc', '#1f77b4', '#ff7f0e', '#2ca02c']
    )

    plt.legend(loc='upper left')
    plt.title("Decomposition of Sales into Base and Media Contributions")
    plt.xlabel("Week")
    plt.ylabel("Sales")

    return fig

def evaluate_model(trace, var_names=None):
    """
    Generate summary statistics for model parameters.

    Parameters:
    -----------
    trace : InferenceData
        ArviZ InferenceData with posterior samples
    var_names : list
        Variables to include in summary

    Returns:
    --------
    pd.DataFrame
        Summary statistics for model parameters
    """
    if var_names is None:
        var_names = [
            'intercept', 'beta_trend', 'beta_sin', 'beta_cos',
            'beta_tv', 'beta_dig', 'beta_print', 'sigma'
        ]

    summary = az.summary(
        trace,
        var_names=var_names,
        hdi_prob=0.95
    )

    return summary

def main():
    """Main execution function"""
    # Generate synthetic data
    data, true_params = generate_synthetic_data(seed=42, n_weeks=104)
    print("Generated synthetic data:")
    print(data.head())

    # Create and fit model
    model = create_mmm_model(data)
    trace = fit_model(model, seed=[42, 43], chains=2)

    # Evaluate model results
    summary = evaluate_model(trace)
    print("\nModel parameter estimates:")
    print(summary)

    # Compute and visualize contributions
    contributions = compute_contributions(data, trace)

    print("\nMedia Contribution Summary:")
    print(f"Total TV contribution: {contributions['summary']['total_tv']:.1f}")
    print(f"Total Digital contribution: {contributions['summary']['total_digital']:.1f}")
    print(f"Total Print contribution: {contributions['summary']['total_print']:.1f}")

    # Create and display visualization
    fig = plot_contributions(data, contributions)
    plt.show()

if __name__ == "__main__":
    main()