"""
This codes creates a dummy database in csv
"""

import pandas as pd
from ahorro.utils.paths import DATA_DIR, PROJECT_ROOT

users = pd.DataFrame(
    {
        "user_id": [1, 2, 3],
        "username": ["admin", "user1", "user2"],
        "pwd": ["admin", "user1", "user2"],
    }
)
users.to_csv(DATA_DIR / "users_plain.csv", index=False)

clients = pd.DataFrame(
    {
        "client_id": [1, 2, 3],
        "client_name": ["Admin", "Cliente 1", "Cliente 2"],
        "user_id": [1, 2, 3],
    }
)
clients.to_csv(DATA_DIR / "clients.csv", index=False)
