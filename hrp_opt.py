import streamlit as st 
from pypfopt.expected_returns import returns_from_prices
from pypfopt.hierarchical_portfolio import HRPOpt


def hrp_screen(df):

    hist_rets = returns_from_prices(df)

    hp = HRPOpt(hist_rets, cov_matrix=None)

    link_method = st.selectbox('Change linkage method (optional):', ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'), index=0, help='''
    Methods for calculating the distance between the newly formed cluster u and each v.\n
    single -> Nearest Point Algorithm;\n
    complete -> Farthest Point Algorithm or Voor Hees Algorithm;\n
    average -> UPGMA algorithm;\n
    weighted -> WPGMA;\n
    centroid -> UPGMC algorithm;\n
    median -> WPGMC algorithm;\n
    ward -> Ward variance minimization algorithm.\n
    ''')

    w = hp.optimize(linkage_method=link_method)

    performance = hp.portfolio_performance()

    parameters_desc = f"""
    Linkage Method: {link_method}. 
    """


    return w, performance, hp, parameters_desc