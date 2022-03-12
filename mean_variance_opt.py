import streamlit as st 
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt.expected_returns import mean_historical_return, ema_historical_return, capm_return
from pypfopt.risk_models import risk_matrix, sample_cov, semicovariance, exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions, plotting


def plot_cov_heatmap(df):
    fig = plt.figure()
    sns.heatmap(df, annot=True)
    plt.title('Covariance Matrix')
    plt.close()
    return fig




def mean_variance_screen(df, sector_mapper):

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

    with st.expander('Show Expected Returns'):
        st.write(mu)

    st.write(' ')
    # covariance matrix
    risk_m = st.selectbox('Covariance Matrix Risk Model:', ('Sample Covariance', 'Semicovariance', 'Exponentially-Weighted Covariance', 'Ledoit-Wolf', 'Ledoit-Wolf Constant Correlation', 'Ledoit-Wolf Single Factor', 'Oracle Approximating'), help='''
    Sample Covariance -> calculate the annualised sample covariance matrix of (daily) asset returns;\n
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
            fix_m = st.selectbox('Covariance matrix fix method:', ('spectral', 'diag'))
            S = semicovariance(df, benchmark=((1+(drf/100))**(1/252)-1), fix_method=fix_m)
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

    with st.expander('Show Covariance Matrix'):
        st.write(plot_cov_heatmap(S))
        
        
        


    ef = EfficientFrontier(mu, S)

    st.write(' ')
    reg = st.checkbox('Use L2 Regularization', help='''
    Mean-variance optimization often sets many of the asset weights to be zero. This may not be ideal if you need to have a certain number of positions in your portfolio, for diversification purposes or otherwise. By adding L2 regularization, you can encourage the optimizer to choose different weights.\n
    It has exactly the same form as the L2 regularization term in machine learning, though a slightly different purpose (in ML it is used to keep weights small while here it is used to make them larger).\n
    ''')
    if reg:
        reg_param = st.slider('Choose reularization parameter:', min_value=0.0, value=0.1, max_value=10.0)
        ef.add_objective(objective_functions.L2_reg, gamma=reg_param)


    sector_constr = st.checkbox('Add sector constraints')
    if sector_constr:
        sector_map = sector_mapper
        sectors = set(sector_map.values())
        sector_lower = {}
        sector_upper = {}
        for sector in sectors:
            st.write(f"{sector} :")
            col1, col2 = st.columns(2)
            with col1:
                lower = st.slider("Min Exposure", min_value=0.0, max_value=1.0, value=0.0, key=sector)
            with col2:
                upper = st.slider("Max Exposure", min_value=0.0, max_value=1.0, value=1.0, key=sector)
            sector_lower[sector] = lower
            sector_upper[sector] = upper

        ef.add_sector_constraints(sector_map, sector_lower, sector_upper)


    if exp_ret_m != 'Exponentially-Weighted Mean':
        ef1 = copy.deepcopy(ef)
        fig2, ax = plt.subplots()
        plotting.plot_efficient_frontier(ef1, ax=ax, ef_param='return')
        plt.close()

    st.write(' ')
    opt_goal = st.selectbox('Optimization Goal:', ('Min Volatility', 'Max Sharpe', 'Max Quadratic Utility', 'Efficient Risk', 'Efficient Return'), 
        help='''
            Min Volatility -> optimizes for minimum volatility;\n
            Max Sharpe -> optimizes for max Sharpe Ratio (tangency portfolio);\n
            Max Quadratic Utility -> maximizes the quadratic utility, given some risk aversion;\n
            Efficient Risk -> maximize return for a target risk;\n
            Efficient Return -> minimize volatility for a given target return.\n
        ''')

    if opt_goal == 'Min Volatility':
        w = ef.min_volatility()
        cw = ef.clean_weights()
        if exp_ret_m != 'Exponentially-Weighted Mean':
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")
            ax.legend()

    elif opt_goal == 'Max Sharpe':
        rf = st.number_input('Change risk free rate (%) :', min_value=0.0, value=2.0)
        w = ef.max_sharpe(risk_free_rate=rf/100)
        cw = ef.clean_weights()
        if exp_ret_m != 'Exponentially-Weighted Mean':
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
            ax.legend()

    elif opt_goal == 'Max Quadratic Utility':
        risk_ave = st.number_input('Risk aversion:', min_value=0.01, value=1.0)
        w = ef.max_quadratic_utility(risk_aversion=risk_ave)
        cw = ef.clean_weights()
        if exp_ret_m != 'Exponentially-Weighted Mean':
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Quadratic Utility")
            ax.legend()

    elif opt_goal == 'Efficient Risk':
        target_vol = st.number_input('Add annual target volatility (%) :', min_value=0.01, value=20.0, help='If you pass an unreasonable target the optimizer will fail silently and return weird weights. Caveat emptor applies!')
        w = ef.efficient_risk(target_vol/100)
        cw = ef.clean_weights()
        if exp_ret_m != 'Exponentially-Weighted Mean':
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Efficient Risk")
            ax.legend()

    elif opt_goal == 'Efficient Return':
        target_ret = st.number_input('Add annual target return (%) :', min_value=0.01, value=20.0, help='If you pass an unreasonable target the optimizer will fail silently and return weird weights. Caveat emptor applies!')
        w = ef.efficient_return(target_ret/100)
        cw = ef.clean_weights()
        if exp_ret_m != 'Exponentially-Weighted Mean':
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Efficient Return")
            ax.legend()

    if exp_ret_m != 'Exponentially-Weighted Mean':
        with st.expander('Show Efficient Frontier'):
            st.write(fig2)
    
    performance = ef.portfolio_performance()

    parameters_desc = f"""
    Expected return model: {exp_ret_m}; 
    Risk model: {risk_m}; 
    Optimization goal: {opt_goal}; 
    Sector_constraints: {sector_constr}; 
    L2_schrinkage: {reg}.
    """

    


    return cw, performance, parameters_desc



       