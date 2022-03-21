import config 
import asyncio
import asyncpg

import yfinance as yf


from datetime import datetime, timedelta

import time 


async def write_to_db(connection, params):
    await connection.execute('''
        CREATE TEMPORARY TABLE p_data (
            asset_id INTEGER,
            dt TIMESTAMP WITHOUT TIME ZONE,
            adj_close NUMERIC NOT NULL
            )
    ''')
    await connection.copy_records_to_table('p_data', records=params)
    await connection.execute('''
        INSERT INTO {table} (asset_id, dt, adj_close)
        SELECT * FROM p_data
        ON CONFLICT (asset_id, dt)
        DO UPDATE SET adj_close=EXCLUDED.adj_close
        WHERE {table}.adj_close <> EXCLUDED.adj_close
    '''.format(table='asset_price_test5')) 
    await connection.execute('''
        DROP TABLE p_data
    ''')
    
    
    

async def get_price(pool, asset_id, symbol):
    try:
        async with pool.acquire() as connection:   
            price_data = yf.download(symbol, start='2000-01-01', end=(datetime.today()-timedelta(days=1)).strftime("%Y-%m-%d"), interval='1d', threads=True)
            close_prices = price_data['Adj Close'].reset_index()
            params = [(asset_id, close_prices['Date'][rec], close_prices['Adj Close'].values[rec]) for rec in range(len(close_prices))]
            await write_to_db(connection, params)
            print(f"Updated {symbol} daily close prices added to db.")
    except Exception as e:
        print(f"Unable to get {symbol} data due to {e.__class__}")



async def get_prices(pool, symbols):
    try:
        ret = await asyncio.gather(*[get_price(pool, asset_id, symbols[asset_id]) for asset_id in symbols])
        print(f"Finalized all. Returned list of {len(ret)} outputs.")
    except Exception as e:
        print(e)



async def get_assets():
    
    pool = await asyncpg.create_pool(host=os.getenv('DB_HOST'), database=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'), command_timeout=60)
    
    async with pool.acquire() as connection:
        assets = await connection.fetch("SELECT * FROM asset WHERE id IN (SELECT holding_id FROM portfolio_holding)")
    
        symbols = {}   
        for asset in assets:
            symbols[asset['id']] = asset['symbol']

    
    await get_prices(pool, symbols)
        
        


start = time.time()
                            
asyncio.get_event_loop().run_until_complete(get_assets())

end = time.time()

print("Took {} seconds".format(end-start))