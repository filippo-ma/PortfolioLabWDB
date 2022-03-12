import streamlit as st 
from pypfopt.expected_returns import mean_historical_return, ema_historical_return, capm_return, returns_from_prices
from pypfopt import objective_functions, EfficientCDaR


def efficient_cdar_screen(df, sector_mapper):

    st.write(' ')
    # returns
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


    # historical returns
    hist_rets = returns_from_prices(df)

    
    ecd = EfficientCDaR(mu, hist_rets)

    st.write(' ')
    reg = st.checkbox('Use L2 Regularization', help='''
    Mean-variance optimization often sets many of the asset weights to be zero. This may not be ideal if you need to have a certain number of positions in your portfolio, for diversification purposes or otherwise. By adding L2 regularization, you can encourage the optimizer to choose different weights.\n
    It has exactly the same form as the L2 regularization term in machine learning, though a slightly different purpose (in ML it is used to keep weights small while here it is used to make them larger).\n
    ''')
    if reg:
        reg_param = st.slider('Choose reularization parameter:', min_value=0.0, value=0.1, max_value=10.0)
        ecd.add_objective(objective_functions.L2_reg, gamma=reg_param)


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

        ecd.add_sector_constraints(sector_map, sector_lower, sector_upper)

    st.write(' ')
    opt_goal = st.selectbox('Optimization Goal:', ('Min CDaR', 'Efficient Risk', 'Efficient Return'), 
        help='''
            Min CDaR -> minimises the CDaR;\n
            Efficient Risk -> maximises return for a given target CDaR;\n
            Efficient Return -> minimises CDaR for a given target return.\n
        ''')

    if opt_goal == 'Min CDaR':
        w = ecd.min_cdar()
        cw = ecd.clean_weights()

    elif opt_goal == 'Efficient Risk':
        targ_cdar = st.number_input('Add desired maximum CDaR of the resulting portfolio (%) :', min_value=1.0, value=30.0, help='Min = Min CDaR; If you pass an unreasonable target the optimizer will fail silently and return weird weights. Caveat emptor applies!')
        w = ecd.efficient_risk(target_cdar=(targ_cdar/100))
        cw = ecd.clean_weights()

    elif opt_goal == 'Efficient Return':
        target_ret = st.number_input('Add annual target return (%) :', min_value=0.01, value=20.0, help='If you pass an unreasonable target the optimizer will fail silently and return weird weights. Caveat emptor applies!')
        w = ecd.efficient_return(target_ret/100)
        cw = ecd.clean_weights()

    performance = ecd.portfolio_performance() 


    parameters_desc = f"""
    Expected return model: {exp_ret_m}; 
    Optimization goal: {opt_goal}; 
    Sector_constraints: {sector_constr}; 
    L2_schrinkage: {reg}.
    """


    return cw, performance, parameters_desc