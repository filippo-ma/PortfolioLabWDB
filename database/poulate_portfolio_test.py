import config
import asyncio

import yfinance as yf
import psycopg2
import psycopg2.extras

#import getassetdata

from datetime import datetime 


connection = psycopg2.connect(host=os.getenv('DB_HOST'), database=os.getenv('DB_NAME'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))

cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)


portfolio_test1_list = ['MSFT', 'JPM', 'LMT', 'TSLA', 'AAPL', 'SBUX', 'AMZN', 'ABT', 'V', 'PYPL']

portfolio_test2_list = ['MSFT', 'FNB', 'LMT', 'TSLA', 'AAPL', 'SBUX', 'AMZN', 'ABT', 'V']

portfolio_test3_list = ['MSFT', 'FNB', 'TWLO', 'TSLA', 'E', 'SBUX', 'AMZN', 'ABT', 'V']

dt_today = datetime.now()




def add_new_portfolio_holding(portfolio_name, holding_list):
    creation_date = datetime.now()

    cursor.execute("""
        SELECT * FROM portfolio WHERE portfolio_name = %s
    """, (portfolio_name,))
    portfolio = cursor.fetchone()

    if portfolio: 
        for holding in holding_list:

            cursor.execute("""
                SELECT * FROM asset WHERE symbol = %s
            """, (holding,))
            asset = cursor.fetchone()
            if asset:
                cursor.execute("""
                    INSERT INTO portfolio_holding (portfolio_id, holding_id, added_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (portfolio_id, holding_id, added_at) DO NOTHING
                """, (portfolio['id'], asset['id'], creation_date))
                connection.commit()
                print(f"{holding} ({asset['name']}) added to {portfolio['portfolio_name']} portfolio")
            else:
                print(f"{holding} not in app database, I'll search in yfinance")

                getassetdata.insert_asset_into_db([holding])

                cursor.execute("""
                    SELECT * FROM asset WHERE symbol = %s
                """, (holding,))
                asset = cursor.fetchone()
                if asset:

                    cursor.execute("""
                    INSERT INTO portfolio_holding (portfolio_id, holding_id, added_at)
                    VALUES (%s, %s, %s)
                    """, (portfolio['id'], asset['id'], creation_date))
                    connection.commit()
                    print(f"{holding} ({asset['name']}) added to {name} portfolio.")
                else:
                    print(f"{holding} doesn't added to {name} portfolio.") 
    else:
        print('Portfolio not found.') 






def add_portfolio(portfolio_user, portfolio_name, portfolio_holding_list):
    creation_date = datetime.now()
    name = portfolio_name
    p_user = portfolio_user

    cursor.execute("""
        insert INTO portfolio (portfolio_name, created_at, portuser)
        VALUES (%s, %s, %s)
        ON CONFLICT (portfolio_name) DO NOTHING
    """, (name, creation_date, p_user))
    connection.commit()
    print(f"Success: {name} portfolio created.")


    cursor.execute("""
        SELECT * FROM portfolio WHERE portfolio_name = %s
    """, (name,))
    portfolio = cursor.fetchone() 

    for holding in portfolio_holding_list:

        cursor.execute("""
            SELECT * FROM asset WHERE symbol = %s
        """, (holding,))
        asset = cursor.fetchone()
        if asset:
            cursor.execute("""
                INSERT INTO portfolio_holding (portfolio_id, holding_id, added_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (portfolio_id, holding_id, added_at) DO NOTHING
            """, (portfolio['id'], asset['id'], creation_date))
            connection.commit()
            print(f"{holding} ({asset['name']}) in {name} portfolio")
        else:
            print(f"{holding} not in app database, I'll search in yfinance")

            getassetdata.insert_asset_into_db([holding])

            cursor.execute("""
                SELECT * FROM asset WHERE symbol = %s
            """, (holding,))
            asset = cursor.fetchone()

            if asset:

                cursor.execute("""
                    INSERT INTO portfolio_holding (portfolio_id, holding_id, added_at)
                    VALUES (%s, %s, %s)
                """, (portfolio['id'], asset['id'], creation_date))
                connection.commit()

                print(f"{holding} ({asset['name']}) in {name} portfolio.")
            else:
                print(f"{holding} doesn't added to {name} portfolio.")
