import streamlit as st 
from pypfopt.expected_returns import mean_historical_return, ema_historical_return, capm_return
from pypfopt.risk_models import risk_matrix, sample_cov, semicovariance, exp_cov
from pypfopt.cla import CLA
from pypfopt import plotting
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cov_heatmap(df):
    fig = plt.figure()
    sns.heatmap(df, annot=True)
    plt.title('Covariance Matrix')
    plt.close()
    return fig

def cla_screen(df):

    st.write(' ')
    # exp returns
    exp_ret_m = st.selectbox('Expected Returns Model:', ('Mean Historical Return', 'Exponentially-Weighted Mean', 'CAPM'), help='''
    mean historical return -> Calculate annualized mean (daily) historical return from input (daily) asset prices;\n
    exponentially-weighted mean -> Calculate the exponentially-weighted mean of (daily) historical returns, giving higher weight to more recent data;\n
    CAPM -> Compute a return estimate using the Capital Asset Pricing Model.\n
    ''')
    if exp_ret_m == 'Mean Historical Return':
        with st.expander('Mean historical return optional parameters:'):
            arith_mean = st.checkbox('Use arithmetic mean instead of geometric mean (CAGR)')
            if arith_mean:
                mu = mean_historical_return(df, compounding=False)
            else:
                mu = mean_historical_return(df, compounding=True)

    elif exp_ret_m == 'Exponentially-Weighted Mean':
        with st.expander('Exponentially-weighted mean optional parameters:'):
            spa = st.slider('Time-span of the EMA (for decay)', value=500, min_value=1, max_value=1000, help='decay=2/(span+1)')
            arith_mean = st.checkbox('Use arithmetic mean instead of geometric mean (CAGR)')
            if arith_mean:
                mu = ema_historical_return(df, span=spa, compounding=False)
            else:
                mu = ema_historical_return(df, span=spa, compounding=True)

    elif exp_ret_m == 'CAPM':
        with st.expander('CAPM optional parameters:'):
            rf = st.number_input('Risk free rate (%) :', min_value=0.001, value=2.0)
            arith_mean = st.checkbox('Use arithmetic mean instead of geometric mean (CAGR)')
            if arith_mean:
                mu = capm_return(df, risk_free_rate=rf/100, compounding=False)
            else:
                mu = capm_return(df, risk_free_rate=rf/100, compounding=True)
    
    with st.expander('Show expected returns'):
        st.write(mu)

    st.write(' ')
    # covariance matrix
    risk_m = st.selectbox('Covariance Matrix Risk Model:', ('Sample Covariance', 'Semicovariance', 'Exponentially-Weighted Covariance', 'Ledoit-Wolf', 'Ledoit-Wolf Constant Correlation', 'Ledoit-Wolf Single Factor', 'Oracle Approximating'), help='''
    Sample Covariance -> calculate the annualized sample covariance matrix of (daily) asset returns;\n
    Semicovariance -> estimate the semicovariance matrix, i.e the covariance given that the returns are less than the benchmark (The semivariance is the variance of all returns which are below some benchmark, typically the risk-free rate. It is a common measure of downside risk.);\n
    Exponentially-Weighted Covariance -> estimate the exponentially-weighted covariance matrix, which gives greater weight to more recent data.\n

    Shrinkage methods (shrink the sample covariance matrix towards the shrinkage target):\n 
    Ledoit-Wolf -> the shrinkage target is the diagonal matrix with the mean of asset variances on the diagonals and zeros elsewhere;\n
    Ledoit-Wolf Constant Correlation -> all pairwise correlations are set to the average correlation (sample variances are unchanged);\n
    Ledoit-Wolf Single Factor -> based on Sharpe's single-index model which effectively uses a stocks's beta to the market as a risk model;\n

    Oracle Approximating Shrinkage (OAS) -> has a lower mean-squared error than Ledoit-Wolf shrinkage when samples are Gaussian or near-Gaussian, Chen et al. (2010).\n
    ''')
    if risk_m == 'Sample Covariance':
        S = sample_cov(df)
    elif risk_m == 'Semicovariance':
        with st.expander('Semicovariance optional parameters:'):
            drf = st.number_input('Daily risk-free rate (%) :', min_value=0.1, value=2.0)
            S = semicovariance(df, benchmark=((1+(drf/100))**(1/252)-1))
    elif risk_m == 'Exponentially-Weighted Covariance':
        with st.expander('Exponentially-weighted covariance optional parameters:'):
            sp = st.slider('Time-span of the EMA (for decay)', value=180, min_value=1, max_value=1000, help='decay=2/(span+1)')
            S = exp_cov(df, span=sp)
    elif risk_m == 'Ledoit-Wolf':
        S = risk_matrix(df, method='ledoit_wolf')
    elif risk_m == 'Ledoit-Wolf Constant Correlation':
        S = risk_matrix(df, method='ledoit_wolf_constant_correlation')
    elif risk_m == 'Ledoit-Wolf Single Factor':
        S = risk_matrix(df, method='ledoit_wolf_single_factor')
    elif risk_m == 'Oracle Approximating':
        S = risk_matrix(df, method='oracle_approximating')

    with st.expander('Show covariance matrix heatmap'):
        st.write(plot_cov_heatmap(S))


    clap = CLA(mu, S)

    st.write(' ')
    opt_goal = st.selectbox('Optimization Goal:', ('Min Volatility', 'Max Sharpe'), 
        help='''
            Min Volatility -> optimizes for minimum volatility;\n
            Max Sharpe -> optimizes for max Sharpe Ratio (tangency portfolio);\n
        ''')

    if opt_goal == 'Min Volatility':
        w = clap.min_volatility()
        cw = clap.clean_weights()

    elif opt_goal == 'Max Sharpe':
        w = clap.max_sharpe()
        cw = clap.clean_weights()

    fig1 = plt.figure()
    plotting.plot_efficient_frontier(clap)
    plt.close()
    with st.expander('Show efficient frontier'):
        st.write(fig1)
    
    performance = clap.portfolio_performance()

    parameters_desc = f"""
    Expected return model: {exp_ret_m}; 
    Optimization goal: {opt_goal}.
    """



    return cw, performance, parameters_desc