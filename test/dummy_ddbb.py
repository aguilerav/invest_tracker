"""
This codes creates a dummy database in csv
"""

import pandas as pd
from ahorro.utils.paths import DATA_DIR, PROJECT_ROOT

# Users for login
users = pd.DataFrame(
    {
        "user_id": [1, 2, 3],
        "username": ["admin", "user1", "user2"],
        "pwd": ["admin", "user1", "user2"],
    }
)
users.to_csv(DATA_DIR / "users_plain.csv", index=False)

# Clients information
clients = pd.DataFrame(
    {
        "client_id": [1, 2, 3],
        "client_name": ["Admin", "Cliente 1", "Cliente 2"],
        "user_id": [1, 2, 3],
    }
)
clients.to_csv(DATA_DIR / "clients.csv", index=False)

# Tickers
tickers = pd.DataFrame(
    {
        "ticker_id": [1, 2, 3],
        "ticker_name": ["Apple", "Alphabet", "Microsoft"],
        "ticker_symbol": ["AAPL", "GOOGL", "MSFT"],
    }
)
tickers.to_csv(DATA_DIR / "tickers.csv", index=False)

# Transactions information
trx = pd.DataFrame(
    {
        "trx_id": [1, 2, 3, 4, 5],
        "user_id": [1, 1, 1, 2, 3],
        "amount": [200.0, 100.0, 300.0, 50.0, 150.0],
        "type": ["buy", "sell", "buy", "buy", "buy"],
        "date": ["2024-12-01", "2024-12-12", "2024-12-20", "2024-12-10", "2024-12-09"],
        "ticker_id": [1, 1, 2, 3, 1],
    }
)
trx.to_csv(DATA_DIR / "transactions.csv", index=False)
