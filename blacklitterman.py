import streamlit as st 
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
#from datetime import datetime, timedelta


from pypfopt.risk_models import CovarianceShrinkage, risk_matrix, sample_cov, semicovariance, exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import black_litterman, risk_models, plotting, objective_functions, cla



# @st.experimental_singleton
# def fetch_mcaps(symbols):
#     mcaps = {}
#     for symbol in symbols:
#         data = yf.Ticker(symbol)
#         mcaps[symbol] = data.info['marketCap']
    
#     return mcaps


def plot_cov_heatmap(df):
    fig = plt.figure()
    sns.heatmap(df, annot=True)
    plt.title('Black-Litterman Covariance Matrix')
    plt.close()
    return fig


def plot_bl_rets(rprior_df, rpost_df):

    df_prior = pd.DataFrame(rprior_df.copy().rename('rets'))
    df_prior['returns'] = ['Prior']*len(df_prior)
    df_posterior = pd.DataFrame(rpost_df.copy().rename('rets'))
    df_posterior['returns'] = ['Posterior']*len(df_posterior)
    df_rets = pd.concat([df_prior, df_posterior], axis=0)

    fig = plt.figure()
    sns.barplot(data=df_rets, y='rets', x=df_rets.index ,hue='returns')
    plt.title('Black-Litterman Expected Returns')
    plt.close()

    return fig





def bl_all_screen(df, symbols, mcapd):

    market_caps = mcapd

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
   
    

    # bl parameters 
    with st.expander("Black-Litterman parameters:"):

        market_tick = st.text_input('Market benchmark (default=SPY):', value='SPY')
        
        prices_data = yf.download(market_tick)
        market_prices = prices_data['Adj Close']

        delta = black_litterman.market_implied_risk_aversion(market_prices)

        pi_ = st.selectbox('Prior estimate of returns:', ('Use a market-implied prior', 'Use an equal-weighted prior', 'No prior'))

        col1, col2 = st.columns(2)
        with col2:
            st.write(' ')
            st.write(' ')
            mira = st.checkbox('Set equal to market implied risk aversion', value=True)
        with col1:
            if mira:
                risk_ave_par_ = st.number_input('Risk aversion parameter:', value=delta, min_value=0.0)
            else:
                risk_ave_par_ = st.number_input('Risk aversion parameter:', value=1.0, min_value=0.0)


    market_prior = black_litterman.market_implied_prior_returns(market_caps, delta, S)


        
    # views
    view_dict = {}
    confidence_dict = []
    view_list = []
    ps = []
    with st.expander('Add views:', expanded=True):
        col1, col2 = st.columns(2)
        with col1: 
            st.subheader('Views:')
        with col2:
            st.write(' ')
            w_conf = st.checkbox("Add confidence intevals (%)")
        st.write("Add absolute or relative views on the assets. You can also specifify your view uncertainties as percentage confidences. In this case the uncertainty matrix will be estimated via Idzorek's method. Otherwise it will be proportional to asset variance (He and Litterman method, 1999).")
        if w_conf:
            for symbol in symbols:
                st.write(' ')
                add_view = st.checkbox(f"Add {symbol} view:")
                if add_view:
                    view_type = st.radio("type", options=("Absolute view", "Relative view"), key=symbol)
                    if view_type == "Absolute view":
                        col1, col2 = st.columns(2)
                        with col1:
                            view_dict[symbol] = st.number_input(f"Enter {symbol} view:")
                        with col2:
                            confidence = st.slider(f"{symbol} view confidence:", min_value=0.0, max_value=1.0, value=0.75)
                        confidence_dict.append(confidence) 
                        view_list.append(view_dict[symbol])
                        a = np.zeros((len(symbols),))
                        a[symbols.index(f"{symbol}")] = 1
                        ps.append(a)
                    elif view_type == "Relative view":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            t_syms = [item for item in symbols if item != symbol]
                            underper_sym = st.selectbox(f"{symbol} will outperform:", options=t_syms)
                            a = np.zeros((len(symbols),))
                            a[symbols.index(f"{symbol}")] = 1
                            a[symbols.index(f"{underper_sym}")] = -1
                            ps.append(a)
                        with col2:
                            outp_value = st.number_input("By:", key=symbol)
                            view_list.append(outp_value)
                        with col3:
                            confidence = st.slider(f"{symbol} view confidence:", min_value=0.0, max_value=1.0, value=0.75)
                        confidence_dict.append(confidence)
        else:
            for symbol in symbols:
                st.write(' ')
                add_view = st.checkbox(f"Add {symbol} view")
                if add_view: 
                    view_type = st.radio("type", options=("Absolute view", "Relative view"), key=symbol)
                    if view_type == "Absolute view":
                        view_dict[symbol] = st.number_input(f"Enter {symbol} view:")
                        view_list.append(view_dict[symbol])
                        a = np.zeros((len(symbols),))
                        a[symbols.index(f"{symbol}")] = 1
                        ps.append(a)
                    
                    elif view_type == "Relative view":
                        col1, col2 = st.columns(2)
                        with col1:
                            t_syms = [item for item in symbols if item != symbol]
                            underper_sym = st.selectbox(f"{symbol} will outperform:", options=t_syms)
                            a = np.zeros((len(symbols),))
                            a[symbols.index(f"{symbol}")] = 1
                            a[symbols.index(f"{underper_sym}")] = -1
                            ps.append(a)
                        with col2:
                            outp_value = st.number_input("By:", key=symbol)
                            view_list.append(outp_value)

        views_vector = np.array([view_list]).reshape(-1,1) 
        picking_matrix = np.array(ps)

       
    
        
    # compute bl
    if pi_ == 'Use a market-implied prior':
        if len(views_vector) > 0:
            if w_conf:
                bl = BlackLittermanModel(S, pi='market', market_caps=market_caps, risk_aversion=risk_ave_par_, Q=views_vector, P=picking_matrix, view_confidences=confidence_dict, omega='idzorek')
            else:
                bl = BlackLittermanModel(S, pi='market', market_caps=market_caps, risk_aversion=risk_ave_par_, Q=views_vector, P=picking_matrix)
            rets_bl = bl.bl_returns()
            covar_bl = bl.bl_cov()
            w_bl=bl.bl_weights(delta)
            implied_bl_weights = bl.clean_weights()
        else:
            rets_bl = market_prior 
            covar_bl = S

    elif pi_ == 'Use an equal-weighted prior':
        if len(views_vector) > 0:
            if w_conf:
                bl = BlackLittermanModel(S, pi='equal', risk_aversion=delta, Q=views_vector, P=picking_matrix, view_confidences=confidence_dict, omega='idzorek')
            else:
                bl = BlackLittermanModel(S, pi='equal', risk_aversion=delta, Q=views_vector, P=picking_matrix)
            rets_bl = bl.bl_returns()
            covar_bl = bl.bl_cov()
            w_bl=bl.bl_weights(delta)
            implied_bl_weights = bl.clean_weights()
        else:
            rets_bl = market_prior 
            covar_bl = S

    elif pi_ == 'No prior':
        if len(views_vector) > 0:
            if w_conf:
                bl = BlackLittermanModel(S, risk_aversion=delta, Q=views_vector, P=picking_matrix, view_confidences=confidence_dict, omega='idzorek')
            else:
                bl = BlackLittermanModel(S, risk_aversion=delta, Q=views_vector, P=picking_matrix)
            rets_bl = bl.bl_returns()
            covar_bl = bl.bl_cov()
            w_bl=bl.bl_weights(delta)
            implied_bl_weights = bl.clean_weights()
        else:
            rets_bl = market_prior 
            covar_bl = S
            

    # plotting bl
    with st.expander('Show Black-Litterman Returns'):
        st.write(plot_bl_rets(market_prior, rets_bl))

    with st.expander('Show Black-Litterman Covariance Matrix'):
        st.write(plot_cov_heatmap(covar_bl)) 

    if len(views_vector) > 0:
        with st.expander('Show Black-Litterman return-implied weights'):
            st.write(implied_bl_weights)


    # optimize
    st.write(' ') 
    st.write("Input BL posterior estimates of returns and covariance into an optimizer...") 
    ef_selection = st.selectbox('Optimizer:', ('Efficient Frontier', 'Critical Line'))
    if ef_selection == 'Efficient Frontier':
        ef = EfficientFrontier(rets_bl, covar_bl)

        ef1 = copy.deepcopy(ef)
        fig2, ax = plt.subplots()
        plotting.plot_efficient_frontier(ef1, ax=ax, ef_param='return')
        plt.close()
        
        st.write(' ')
        opt_goal = st.selectbox('Optimization Goal:', ('Min Volatility', 'Max Sharpe', 'Max Quadratic Utility', 'Efficient Risk', 'Efficient Return'), 
                    help='''
                        Min Volatility -> optimizes for minimum volatility;\n
                        Max Sharpe -> optimizes for max Sharpe Ratio (tangency portfolio);\n
                        Max Quadratic Utility -> maximises the quadratic utility, given some risk aversion;\n
                        Efficient Risk -> maximize return for a target risk;\n
                        Efficient Return -> minimize volatility for a given target return.\n
                    ''')

        if opt_goal == 'Min Volatility':
            w = ef.min_volatility()
            cw_bl = ef.clean_weights()
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Min Volatility")
            ax.legend()
            
        elif opt_goal == 'Max Sharpe':
            rf = st.number_input('Change risk free rate (%) :', min_value=0.0, value=2.0)
            w = ef.max_sharpe(risk_free_rate=rf/100)
            cw_bl = ef.clean_weights()
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
            ax.legend()

        elif opt_goal == 'Max Quadratic Utility':
            risk_ave = st.number_input('Risk aversion:', min_value=0.01, value=1.0)
            w = ef.max_quadratic_utility(risk_aversion=risk_ave)
            cw_bl = ef.clean_weights()
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Quadratic Utility")
            ax.legend()

        elif opt_goal == 'Efficient Risk':
            target_vol = st.number_input('Annual target volatility (%) :', min_value=0.01, value=20.0, help='If you pass an unreasonable target the optimizer will fail silently and return weird weights. Caveat emptor applies!')
            w = ef.efficient_risk(target_vol/100)
            cw_bl = ef.clean_weights()
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Efficient Risk")
            ax.legend()

        elif opt_goal == 'Efficient Return':
            target_ret = st.number_input('Annual target return (%) :', min_value=0.01, value=20.0, help='If you pass an unreasonable target the optimizer will fail silently and return weird weights. Caveat emptor applies!')
            w = ef.efficient_return(target_ret/100)
            cw_bl = ef.clean_weights()
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Efficient Return")
            ax.legend()

        with st.expander("Show Efficient Frontier"):
            st.write(fig2)
    
    elif ef_selection == 'Critical Line':
        ef = cla.CLA(rets_bl, covar_bl)

        opt_goal = st.selectbox('Optimization Goal:', ('Min Volatility', 'Max Sharpe'), 
                    help='''
                        Min Volatility -> optimizes for minimum volatility;\n
                        Max Sharpe -> optimizes for max Sharpe Ratio (tangency portfolio);\n
                    ''')

        if opt_goal == 'Min Volatility':
            w = ef.min_volatility()
            cw_bl = ef.clean_weights()

        elif opt_goal == 'Max Sharpe':
            w = ef.max_sharpe()
            cw_bl = ef.clean_weights()

        fig1 = plt.figure()
        plotting.plot_efficient_frontier(ef)
        plt.close()
        with st.expander('Show efficient frontier'):
            st.write(fig1)

    

    performance = ef.portfolio_performance() 

    parameters_desc = f"""
    Risk model: {risk_m}; 
    Market benchmark: {market_tick};
    Prior: {pi_};
    Absolute views: {view_dict};
    Num views: {len(views_vector)};
    Confidence intervals: {w_conf};
    Optimizer: {ef_selection};
    Optimization goal: {opt_goal}; 
    """

        
    print(cw_bl)
    print(performance)
        

    return cw_bl, performance, parameters_desc



