import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import copy
import cvxopt
import quantstats as qs


from contextlib import contextmanager, redirect_stdout
from io import StringIO

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import plotly.express as px

from mean_variance_opt import mean_variance_screen
from efficient_semivariance_opt import efficient_semivariance_screen
from efficient_cvar_opt import efficient_cvar_screen
from efficient_cdar_opt import efficient_cdar_screen
from hrp_opt import hrp_screen
from cla_opt import cla_screen
from blacklitterman import bl_all_screen

from pypfopt.plotting import plot_dendrogram
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.expected_returns import returns_from_prices

#from concurrent.futures import ThreadPoolExecutor


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield


@st.experimental_memo
def fetch_prices(ticker):

    data = yf.download(ticker)
    df = data['Adj Close']

    return df



@st.experimental_singleton
def fetch_info(asset):

    data = yf.Ticker(asset)
    asset_info = data.info

    keys = ['symbol', 'longName', 'exchange', 'quoteType']

    if None not in list(map(asset_info.get, keys)):
        
        symbol = asset_info['symbol']
        name = asset_info['longName']
        exchange = asset_info['exchange']
        asset_class = asset_info['quoteType']

        if asset_class == 'ETF' or asset_class == 'MUTUALFUND':
            market_cap = asset_info['totalAssets']
        else:
            market_cap = asset_info['marketCap']

        sector_key = 'sector'
        if asset_info.get(sector_key) is not None:
            sector = asset_info['sector']
        else:
            sector = 'sector undefined'
    
        return symbol, name, exchange, asset_class, sector, market_cap
    
    else:
        return None


st.set_page_config(page_title='PortfolioLab', page_icon='static/icon.ico')

st.title('Welcome to PortfolioLab')
st.markdown("<p>Portfolio optimizer. Built w/ <a href=https://pyportfolioopt.readthedocs.io/en/latest/index.html>PyPortfolioOpt</a> and <a href=https://github.com/ranaroussi/quantstats>QuantStats</a>.</p>", unsafe_allow_html=True)




st.subheader('Add tickers to create a portfolio')
with st.form(key='tick_form'):

    tickers = st.text_input(label='Add asset tickers:', placeholder='Enter tickers', help='Insert space separated symbols; Use Yahoo Finance classification system for identifying instruments.', value="AAPL KO BAC AXP KHC")
    submit_button = st.form_submit_button(label='Add tickers')

holdings = pd.DataFrame(columns=['Symbol', 'Name', 'Exchange', 'Asset Class', 'Sector', 'Market Cap'])



if tickers:
    
    st.subheader('Portfolio holdings:')

    sector_mapper = {}
    symbols = []
    for ticker in tickers.split(' '):
        r = fetch_info(ticker)
        if r != None:
            symbol, name, exchange, asset_class, sector, market_cap = r
            symbols.append(symbol)
            sector_mapper[symbol] = sector

            holdings = holdings.append({
                'Symbol': symbol,
                'Name': name,
                'Exchange': exchange,
                'Asset Class': asset_class,
                'Sector': sector,
                'Market Cap': market_cap,
            }, ignore_index=True)

    st.write(holdings)


    #with ThreadPoolExecutor(max_workers=16) as pool:
        #results = pool.map(fetch_info, tickers.split(' '))
            
    # sector_mapper = {}
    # symbols = []
    # for r in results:
    #     if r is not None:
    #         (symbol, name, exchange, asset_class, sector) = r
    #         symbols.append(symbol)
    #         sector_mapper[symbol] = sector

    #         holdings = holdings.append({
    #             'Symbol': symbol,
    #             'Name': name,
    #             'Exchange': exchange,
    #             'Asset Class': asset_class,
    #             'Sector': sector,
    #         }, ignore_index=True)



    df1 = fetch_prices(symbols)
    df = df1.dropna()

    with st.expander('Adjusted Close Prices'):
        st.write(df)

    latest_prices = get_latest_prices(df)
    asset_returns_df = returns_from_prices(df)
    cum_rets = asset_returns_df.apply(lambda x: x.add(1, fill_value=0).cumprod()-1)
    

    date_ss = str(df.index[0]).split(' ')[0]
    date_ff = str(df.index[-1]).split(' ')[0]
    
    fig_rets = px.line(cum_rets, labels={"variable":"asset", "value":"cumulative return"}, title=f"Cumulative asset returns ({date_ss} - {date_ff})")
   
    with st.expander('Show cumulative returns'):
        st.plotly_chart(fig_rets, use_container_width=True)


    st.write(' ')
    st.subheader('Choose optimization method and parameters')

    backt = st.radio('Backtest portfolio allocation?', ('No', 'Yes'))

    if backt == 'Yes':
        lb_p = st.selectbox('total lookback period:', ('1Y','2Y','3Y','5Y','10Y','20Y','MAX'), index=4)
        if lb_p == 'MAX':
            df_t = df
        elif lb_p == '20Y':
            df_t = df.last('240M')
        elif lb_p == '10Y':
            df_t = df.last('120M')
        elif lb_p == '5Y':
            df_t = df.last('60M')
        elif lb_p == '3Y':
            df_t = df.last('36M')
        elif lb_p == '2Y':
            df_t = df.last('24M')
        elif lb_p == '1Y':
            df_t = df.last('12M')

        proportion = st.number_input("Percentage of data to use for optimization:", value=0.70, help='percentage of total lookback period')
        spl = int(len(df_t)*proportion)
        df = df_t.iloc[:spl, :]
        df_backtest = df_t.iloc[spl:, :]
        
        date_s = str(df.index[0]).split(' ')[0]
        date_f = str(df.index[-1]).split(' ')[0]
        date_s_b = str(df_backtest.index[0]).split(' ')[0]
        date_f_b = str(df_backtest.index[-1]).split(' ')[0]

        st.write(f"Optimization period: {date_s} - {date_f}")
        st.write(f"Backtest period: {date_s_b} - {date_f_b}")

    else:
        lb_p = st.selectbox('lookback period:', ('1Y','2Y','3Y','5Y', '10Y', '20Y','MAX'), index=4)
        if lb_p == 'MAX':
            df = df
        elif lb_p == '20Y':
            df = df.last('240M')
        elif lb_p == '10Y':
            df = df.last('120M')
        elif lb_p == '5Y':
            df = df.last('60M')
        elif lb_p == '3Y':
            df = df.last('36M')
        elif lb_p == '2Y':
            df = df.last('24M')
        elif lb_p == '1Y':
            df = df.last('12M')

        date_s = str(df.index[0]).split(' ')[0]
        date_f = str(df.index[-1]).split(' ')[0]

        st.write(f"period: {date_s} - {date_f}")
   

    with st.expander(f"Show returns correlation matrix ({date_s} - {date_f})"):
        ret_corr_matrix = returns_from_prices(df).corr() 
        fig_corr = plt.figure()
        sns.heatmap(ret_corr_matrix, annot=True, cmap='viridis')
        plt.close()
        st.write(fig_corr)


    st.write(' ')
    st.write(' ')



    opt_model = st.selectbox("Optimization Method:", ('Mean-Variance Optimization', 'Efficient Semivariance', 'Efficient CVaR (conditional value at risk)', 'Efficient CDaR (conditional drawdown at risk)', 'Black-Litterman Allocation', 'Hierarchical Risk Parity (HRP)', 'Critical Line Algorithm (CLA)'), help='''
    Mean-Variance Optimization (MVO) -> Markowitz (1952).\n
    Efficient Semivariance -> Instead of penalizing volatility, mean-semivariance optimization seeks to only penalise downside volatility, since upside volatility may be desirable.\n 
    Efficient CVaR (conditional value at risk) -> It allows for optimization along the mean-CVaR frontier, using the formulation of Rockafellar and Ursayev (2001). The conditional value-at-risk (a.k.a expected shortfall) is a popular measure of tail risk.\n
    Efficient CDaR (conditional drawdown at risk) -> It tries to alleviate the problems with Efficient Semivariance and Efficient CvAr in that it accounts for the timespan of material decreases in value.\n
    Black-Litterman Allocation -> The Black-Litterman model takes a Bayesian approach to asset allocation. Specifically, it combines a prior estimate of returns (for example, the market-implied returns) with views on certain assets, to produce a posterior estimate of expected returns. Using Black-Litterman posterior returns results in much more stable portfolios than using mean-historical return.\n
    Hierarchical Risk Parity (HRP) -> Method developed by Marcos Lopez de Prado, it does not require the inversion of the covariance matrix as with traditional mean-variance optimization, and seems to produce diverse portfolios that perform well out of sample.\n
    Critical Line Algorithm (CLA) -> Bailey and Lopez de Prado (2013). This is a robust alternative to the quadratic solver used to find mean-variance optimal portfolios, that is especially advantageous when we apply linear inequalities. Unlike generic convex optimization routines, the CLA is specially designed for portfolio optimization. It is guaranteed to converge after a certain number of iterations, and can efficiently derive the entire efficient frontier.\n 
    ''')


    if opt_model == 'Mean-Variance Optimization':
        w, performance, parameters = mean_variance_screen(df, sector_mapper)

        exp_ann_ret, ann_vol, sharpe = performance
        st.subheader('Results:')
        with st.expander('Portfolio performance', expanded=True):
            st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
            st.write("Annual Volatility: {:.1%}".format(ann_vol))
            st.write("Sharpe Ratio: {:.2}".format(sharpe))


    elif opt_model == 'Efficient Semivariance':
        w, performance, parameters = efficient_semivariance_screen(df, sector_mapper)

        exp_ann_ret, ann_semidev, sortino = performance
        st.subheader('Results:')
        with st.expander('Portfolio performance', expanded=True):
            st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
            st.write("Annual semi-deviation: {:.1%}".format(ann_semidev))
            st.write("Sortino Ratio: {:.2}".format(sortino))

    
    elif opt_model == 'Efficient CVaR (conditional value at risk)':
        w, performance, parameters = efficient_cvar_screen(df, sector_mapper)

        exp_ann_ret, cond_var = performance
        st.subheader('Results:')
        with st.expander('Portfolio performance', expanded=True):
            st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
            st.write("Conditional Value at Risk: {:.2%}".format(cond_var))
        

    elif opt_model == 'Efficient CDaR (conditional drawdown at risk)':
        w, performance, parameters = efficient_cdar_screen(df, sector_mapper)

        exp_ann_ret, cond_dar = performance
        st.subheader('Results:')
        with st.expander('Portfolio performance', expanded=True):
            st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
            st.write("Conditional Drawdown at Risk: {:.2%}".format(cond_dar))
        

    elif opt_model == 'Hierarchical Risk Parity (HRP)':
        w, performance, hrp, parameters = hrp_screen(df)
        
        fig, ax = plt.subplots()
        sch.dendrogram(hrp.clusters, labels=hrp.tickers, ax=ax, orientation="top")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        with st.expander('Clusters dendrogram:', expanded=True):
            st.pyplot(fig)

        exp_ann_ret, ann_vol, sharpe = performance
        st.subheader('Results:')
        with st.expander('Portfolio performance', expanded=True):
            st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
            st.write("Annual Volatility: {:.1%}".format(ann_vol))
            st.write("Sharpe Ratio: {:.2}".format(sharpe))


    elif opt_model == 'Critical Line Algorithm (CLA)':
        w, performance, parameters = cla_screen(df)

        exp_ann_ret, ann_vol, sharpe = performance
        st.subheader('Results:')
        with st.expander('Portfolio performance', expanded=True):
            st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
            st.write("Annual Volatility: {:.1%}".format(ann_vol))
            st.write("Sharpe Ratio: {:.2}".format(sharpe))


    
    elif opt_model == 'Black-Litterman Allocation':

        if backt == 'Yes':
            with st.expander('Market Caps', expanded=True):
                st.write(f"Insert market caps at {date_f}")
                market_caps_d = {}
                with st.form('market_caps'):
                    for symbol in symbols:
                        market_caps_d[symbol] = st.number_input(f"{symbol} market cap:", value=0)
                    sub_btn = st.form_submit_button(label='Submit market caps')
            if sub_btn:
                w, performance, parameters = bl_all_screen(df, symbols, market_caps_d)
                exp_ann_ret, ann_vol, sharpe = performance
                st.subheader('Results:')
                with st.expander('Portfolio performance', expanded=True):
                    st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
                    st.write("Annual Volatility: {:.1%}".format(ann_vol))
                    st.write("Sharpe Ratio: {:.2}".format(sharpe))
            else:
                st.info('Please submit market caps')
                w = False


        else:
            mcap_df = holdings.set_index('Symbol')['Market Cap']

            market_caps_d = mcap_df.to_dict()

            if any(pd.isnull(mc) for mc in market_caps_d.values()):
                with st.expander('Market Caps', expanded=True):
                    st.write(f"Can't find market cap at {date_f} for the following assets, please insert manually")
                    with st.form('m_caps_f'):
                        for asset, mc in market_caps_d.items():
                            if pd.isnull(mc):
                                market_caps_d[asset] = st.number_input(f"{asset} market cap:", value=0)
                        sub_btn = st.form_submit_button(label='Submit market caps')

                w, performance, parameters = bl_all_screen(df, symbols, market_caps_d)
                exp_ann_ret, ann_vol, sharpe = performance
                st.subheader('Results:')
                with st.expander('Portfolio performance', expanded=True):
                    st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
                    st.write("Annual Volatility: {:.1%}".format(ann_vol))
                    st.write("Sharpe Ratio: {:.2}".format(sharpe))
            else:
                w, performance, parameters = bl_all_screen(df, symbols, market_caps_d)
                exp_ann_ret, ann_vol, sharpe = performance
                st.subheader('Results:')
                with st.expander('Portfolio performance', expanded=True):
                    st.write("Expected annual return: {:.1%}".format(exp_ann_ret))
                    st.write("Annual Volatility: {:.1%}".format(ann_vol))
                    st.write("Sharpe Ratio: {:.2}".format(sharpe))





    if w:

        # portfolio weights pie chart     
        w_df = pd.DataFrame(columns=['position', 'weight'], data=w.items())
        fig = px.pie(w_df, values='weight', names='position', title='Optimized Portfolio Weights')
        st.plotly_chart(fig, use_container_width=True)


        # metrics 
        with st.expander('Metrics:'):
            bench_m = st.text_input('Benchmark:', value='SPY')

            opt_rets = returns_from_prices(df)

            start_d_m = df.index[0] + timedelta(days=1)
            end_d_m = df.index[-1] + timedelta(days=1)
            bench_rets_m = returns_from_prices(yf.download(bench_m, start=start_d_m, end=end_d_m)['Adj Close']).rename(f"{bench_m}")
            ports_index = qs.utils.make_index(w, rebalance=None, returns=opt_rets).rename('Portfolio')

            opt_tot_df = pd.concat([ports_index, bench_rets_m], axis=1).dropna()

            col1, col2 = st.columns((4,2))
            with col1:
                output_m = st.empty()
                with st_capture(output_m.code):
                    qs.reports.metrics(opt_tot_df.iloc[:,0], opt_tot_df.iloc[:,1])
            with col2:
                gen_report = st.button("Generate Full Report")
                if gen_report:
                    qs.reports.html(opt_tot_df.iloc[:,0], benchmark=opt_tot_df.iloc[:,1], download_filename='report_opt.html', output=".\output_report\test1.html", title=f"My Portfolio vs {bench_m}")
                    with open("report_opt.html", 'r', encoding='utf-8') as file:
                        btn = st.download_button(label='Download Report', data=file, file_name='report_metrics.html')
                        




        # sector exposure
        temp_df1 = w_df.set_index('position')
        temp_df2 = holdings.set_index('Symbol')
        new_df = temp_df1.join(temp_df2)
        new_df = new_df[['weight', 'Sector']]
        sector_df = new_df.groupby('Sector').sum()
        fig1 = px.pie(sector_df, values='weight', names=sector_df.index, title='Sector Exposure')
        with st.expander("Show Portfolio Sector Exposure"):
            st.plotly_chart(fig1, use_container_width=True)

    

        # portfolio weights comparison
        if 'pfs_w_list' not in st.session_state:
            st.session_state.pfs_w_list = []
        
        with st.expander("Compare portfolio weights"):
            pf_name = st.text_input('Add a name for the current portofolio (required):', placeholder='Enter name')
            description = st.text_input('Add a description (optional):', value=f"Optimization method: {opt_model}; {parameters}; assets:{symbols}")
            btn1 = st.button('Add to comparison list', help="the assets for each portfolio should be the same", key='ass')
            if btn1:
                if len(st.session_state.pfs_w_list) == 0:
                    st.session_state.pfs_w_list.append({'name': pf_name, 'description': description,'weights': w}) 
                    st.success(f"{pf_name} added to list")
                else:
                    for i in range(len(st.session_state.pfs_w_list)):
                        if st.session_state.pfs_w_list[i]['name'] == pf_name:
                            st.warning("Choose another name for the current portfolio")
                        else:
                            st.session_state.pfs_w_list.append({'name': pf_name, 'description': description,'weights': w}) 
                            st.success(f"{pf_name} added to list")

            pfs_w_copy = copy.deepcopy(st.session_state.pfs_w_list)
            
            df_columns = []
            df_data = []
            df_description = []
            for x in pfs_w_copy:
                df_columns.append(x['name'])
                df_data.append(x['weights'])
                df_description.append(x['description'])

            df_compa = pd.DataFrame(df_data).transpose()
            df_compa.columns = df_columns

            df_desc = pd.DataFrame(df_description, columns=['Description'])
            df_desc = df_desc.transpose()
            df_desc.columns = df_columns
            df_desc = df_desc.transpose()

            if len(df_compa) > 0:
                st.write(df_compa)
                st.write(df_desc)

                fig, ax = plt.subplots()
                ax = sns.heatmap(df_compa, annot=True, cmap='cool')
                st.pyplot(fig)

            del pfs_w_copy
            del df_columns, df_data, df_description

    
        
    
    
        # discrete allocation
        with st.expander('Discrete Allocation:'):

            st.write('Convert the current portfolio weights into a number of shares that can be purchased at the latest provided price.')

            port_value = st.number_input('Insert portfolio total value:', value=10000, help="Assume that portfolio value and asset prices are expressed in the same currency.")

            all_method = st.selectbox('Allocation method:', ('Greedy algorithm', 'Integer programming'))

            disc_all_portfolio = DiscreteAllocation(weights=w, latest_prices=latest_prices, total_portfolio_value=port_value)

            if all_method == 'Greedy algorithm':
                dp = disc_all_portfolio.greedy_portfolio()
                dp_rmse = disc_all_portfolio._allocation_rmse_error()

            elif all_method == 'Integer programming':
                dp = disc_all_portfolio.lp_portfolio(solver='GLPK_MI')
                dp_rmse = disc_all_portfolio._allocation_rmse_error()

            n_shares, funds_remaining = dp

            disc_df = pd.DataFrame(n_shares.values(), index=n_shares.keys(), columns=['# of shares'] )

            col1, col2 = st.columns(2)
            with col1:
                st.write(disc_df)
            with col2:
                st.write("Unallocated funds: {:.5}".format(funds_remaining))
                st.write("Allocation RMSE: {:.1}".format(dp_rmse))


            output = st.empty()
            with st_capture(output.code):
                disc_all_portfolio._allocation_rmse_error()



        # backtesting
        if backt == 'Yes':

            def make_index_my(ticker_weights, rebalance="1M", period="max", returns=None, match_dates=False):
                """
                Makes an index out of the given tickers and weights.
                Optionally you can pass a dataframe with the returns.
                If returns is not given it try to download them with yfinance
                Args:
                    * ticker_weights (Dict): A python dict with tickers as keys
                        and weights as values
                    * rebalance: Pandas resample interval or None for never
                    * period: time period of the returns to be downloaded
                    * returns (Series, DataFrame): Optional. Returns If provided,
                        it will fist check if returns for the given ticker are in
                        this dataframe, if not it will try to download them with
                        yfinance
                Returns:
                    * index_returns (Series, DataFrame): Returns for the index
                """
                # Declare a returns variable
                index = None
                portfolio = {}

                # Iterate over weights
                for ticker in ticker_weights.keys():
                    if (returns is None) or (ticker not in returns.columns):
                        # Download the returns for this ticker, e.g. GOOG
                        ticker_returns = download_returns(ticker, period)
                    else:
                        ticker_returns = returns[ticker]

                    portfolio[ticker] = ticker_returns

                # index members time-series
                index = pd.DataFrame(portfolio).dropna()

                if match_dates:
                    index=index[max(index.ne(0).idxmax()):]

                # no rebalance?
                if rebalance is None:
                    for ticker, weight in ticker_weights.items():
                        index[ticker] = weight * index[ticker]
                    return index.sum(axis=1)

                last_day = index.index[-1]

                # rebalance marker
                rbdf = index.resample(rebalance).first()
                rbdf['break'] = rbdf.index.strftime('%s')

                # index returns with rebalance markers
                index = pd.concat([index, rbdf['break']], axis=1)

                # mark first day day
                index['first_day'] = pd.isna(index['break']) & ~ pd.isna(index['break'].shift(1))
                index.loc[index.index[0], 'first_day'] = True

                # multiply first day of each rebalance period by the weight
                for ticker, weight in ticker_weights.items():
                    index[ticker] = np.where(
                        index['first_day'], weight * index[ticker], index[ticker])
                    index[ticker] = weight * index[ticker]

                # drop first marker
                index.drop(columns=['first_day'], inplace=True)

                # drop when all are NaN
                index.dropna(how="all", inplace=True)
                return index[index.index <= last_day].sum(axis=1)


            rets_bt = returns_from_prices(df_backtest).dropna()
            
            with st.expander('Backtesting', expanded=True):
                back_type = st.selectbox('Available Rebalancing Strategies:', ('One-time-allocation (no rebalancing)', 'Calendar Rebalancing (constant weights)'), disabled=True)
                if back_type == 'Calendar Rebalancing (constant weights)':
                    reb_period = st.selectbox('Select rebalancing period', ('1M', '3M', '6M', '1Y'), index = 1)
                    port_w = make_index_my(w, rebalance=reb_period, returns=rets_bt).rename('Portfolio')
                elif back_type == 'One-time-allocation (no rebalancing)':
                    port_w = qs.utils.make_index(w, rebalance=None, returns=rets_bt).rename('Portfolio')
                
                risk_free = st.number_input('Change risk-free rate (optional):', value=0.0)
                
                bench_t = st.selectbox('Benchmark:', ('Use ticker', 'Use lazy weights'))
                if bench_t == 'Use ticker':
                    benchmark = st.text_input('Choose benchmark:', value='SPY')
                    start_d = df_backtest.index[0] + timedelta(days=1)
                    end_d = df_backtest.index[-1] + timedelta(days=1)
                    bench_rets = returns_from_prices(yf.download(benchmark, start=start_d, end=end_d)['Adj Close']).rename(f"{benchmark}")
                    
                elif bench_t == 'Use lazy weights':
                    lazy_w = {}
                    with st.form('weights_form'):
                        st.write('Choose weights:')
                        for symbol in symbols:
                            col1, col2 = st.columns(2)
                            with col1:
                                lazy_w[symbol] = st.number_input(f"{symbol} weight:", value=(1/len(symbols)))
                        submitted = st.form_submit_button("Submit Weights")
                        if submitted:
                            if sum(lazy_w.values()) != 1:
                                st.warning("Weights must sum to 1")
                            else:
                                if back_type == 'Calendar Rebalancing (constant weights)':
                                    bench_rets = make_index_my(lazy_w, rebalance=reb_period, returns=rets_bt).rename('Lazy weights portfolio')
                                    benchmark = f"Lazy weights portfolio ({reb_period} rebalancing)"       
                                elif back_type == 'One-time-allocation (no rebalancing)': 
                                    bench_rets = qs.utils.make_index(lazy_w, rebalance=None, returns=rets_bt).rename('Lazy weights portfolio')
                                    benchmark = 'Lazy weights portfolio (no rebalancing)'
                try:
                    new_df = pd.concat([port_w, bench_rets], axis=1).dropna()
                    cum_rets_ba = new_df.apply(lambda x: (((x+1).cumprod())-1))

                    fig_back = px.line(cum_rets_ba, labels={"value":"cumulative return"}, title=f"Portfolio vs {benchmark} cumulative returns")
                    st.plotly_chart(fig_back, use_container_width=True)

                    col1, col2 = st.columns((3,2))
                    with col1:
                        output = st.empty()
                        with st_capture(output.code):
                            qs.reports.metrics(new_df.iloc[:, 0], benchmark=new_df.iloc[:,1], rf=risk_free)
                    with col2:
                        gen_report = st.button("Generate Full Report", key='full_rep')
                        if gen_report:
                            qs.reports.html(port_w, rf=risk_free, benchmark=bench_rets, download_filename='bt_report.html' , output=".\output_report\test1.html", title=f"My Portfolio vs {benchmark}")
                            with open("bt_report.html", 'r', encoding='utf-8') as file:
                                btn = st.download_button(label='Download Report', data=file, file_name='report.html')     
                except:
                    st.warning('Waiting for weights...')
            
st.write(' ')
st.write(' ')  
st.write(' ')  

st.markdown("""
#### DISCLAIMER\n  
The content of this webpage is not an investment advice and does not constitute any offer or solicitation to offer or recommendation of any investment product. It is for general purposes only and does not take into account your individual needs, investment objectives and specific financial circumstances. Investing involves risk, including possible loss of principal. **Past performance is not a guarantee of future results**. Hypothetical returns do not reflect trading costs, transaction fees, commissions, or actual taxes due on investment returns.
""")


    






    



    

    




  









