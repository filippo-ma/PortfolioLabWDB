CREATE TABLE asset (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    sector TEXT,
    UNIQUE (symbol)
);

CREATE TABLE asset_price (
    asset_id INTEGER NOT NULL,
    dt TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    adj_close NUMERIC NOT NULL,
    PRIMARY KEY (asset_id, dt),
    CONSTRAINT fk_asset FOREIGN KEY (asset_id) REFERENCES asset (id)
);