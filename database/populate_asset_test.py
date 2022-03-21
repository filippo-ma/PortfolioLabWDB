import os
from dotenv import load_dotenv
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import psycopg2
import psycopg2.extras

import streamlit as st
import yfinance as yf

import pandas as pd


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

connection = init_connection()


# dotenv_path = Path('../.env')
# load_dotenv(dotenv_path=dotenv_path)


# connection = psycopg2.connect(host=os.getenv('DB_HOST'), database=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))

cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)


symbols_test = ['AAPL', 'KO']



# def fetch_info(asset):

#     data = yf.Ticker(asset)
#     asset_info = data.info

#     keys = ['symbol', 'longName', 'exchange', 'quoteType']

#     if None not in list(map(asset_info.get, keys)):
        
#         symbol = asset_info['symbol']
#         name = asset_info['longName']
#         exchange = asset_info['exchange']
#         asset_class = asset_info['quoteType']

#         if asset_class == 'ETF' or asset_class == 'MUTUALFUND':
#             market_cap = asset_info['totalAssets']
#         else:
#             market_cap = asset_info['marketCap']

#         sector_key = 'sector'
#         if asset_info.get(sector_key) is not None:
#             sector = asset_info['sector']
#         else:
#             sector = 'sector undefined'
    
#         return symbol, name, exchange, asset_class, sector, market_cap
    
#     else:
#         return None


# def search_asset(symbols):

#     exist_sym = []
#     for symbol in symbols:
    
#         cursor.execute("""
#             SELECT symbol FROM asset WHERE symbol = %s
#         """, (symbol,))

#         ex_sym = cursor.fetchone()

#         if ex_sym:
#             exist_sym.append(ex_sym)

#     if len(exist_sym) > 0:
#         symbols_to_fetch = list(set(symbols)-set(exist_sym))
#     else:
#         symbols_to_fetch = symbols

#     return symbols_to_fetch



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

        st.info(f"{asset} data fetched.")
        return symbol, name, exchange, asset_class, sector, market_cap
    
    else:
        st.warning(f"Impossible to find {asset} data in yfinance, try another ticker.")
        return None


# ast = search_asset(symbols_test)
# print(ast)
#import psycopg2
#import psycopg2.extras



# def insert_asset_into_db(symbols):

#     assets_to_add = search_asset(symbols) 


#     with ThreadPoolExecutor(max_workers=16) as pool:
#         results = pool.map(fetch_info, assets_to_add)
        
#     for r in results:
#         if r is not None:
#             (symbol, name, exchange, asset_class, sector, market_cap) = r

#             cursor.execute("""
#                     INSERT INTO asset (symbol, name, exchange, asset_class, sector)
#                     VALUES (%s, %s, %s, %s, %s)
#                     ON CONFLICT (symbol) DO NOTHING
#             """, (symbol, name, exchange, asset_class, sector))

#             connection.commit()
#             st.success(f"{symbol} ({name}) added to DB.")





def fetch_assets_info(tickers):

    data = {}
    sector_mapper = {}
    for ticker in tickers:
        cursor.execute("""
        SELECT * FROM asset WHERE symbol = %s
        """, (ticker,))
        
        asset_data = cursor.fetchone()
        #print(asset_data)
        if asset_data:
            data[ticker] = asset_data
            sector_mapper[ticker] = asset_data['sector']
            
        else:
            st.info(f"{ticker} data not in DB. ")
            symbol, name, exchange, asset_class, sector, market_cap = fetch_info(ticker)

            cursor.execute("""
                    INSERT INTO asset (symbol, name, exchange, asset_class, sector)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO NOTHING
            """, (symbol, name, exchange, asset_class, sector))

            connection.commit()
            st.success(f"{symbol} ({name}) added to database.")

            cursor.execute("""
            SELECT * FROM asset WHERE symbol = %s
            """, (ticker,))
        
            asset_data = cursor.fetchone()
            data[ticker] = asset_data
            sector_mapper[ticker] = asset_data['sector']

    return data, sector_mapper



def insert_asset_into_db(symbols):

    print('Fetching data from yfinance...')

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = pool.map(get_asset_data, symbols)
        
    for r in results:
        if r is not None:
            (symbol, name, exchange, asset_class, sector) = r

            cursor.execute("""
                    INSERT INTO asset (symbol, name, exchange, asset_class, sector)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO NOTHING
            """, (symbol, name, exchange, asset_class, sector))

            connection.commit()
            print(f"{symbol} ({name}) added to DB.")





# myinfo, sector_mapper = retrieve_assets(symbols_test)

# print(myinfo)
# print(sector_mapper)

# ndf = pd.DataFrame(data=myinfo.values(), columns=['id', 'Symbol', 'Name', 'Exchange', 'Asset Class', 'Sector'])
# print(ndf)
