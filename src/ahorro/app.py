import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash
from pathlib import Path
from datetime import datetime
import csv
from functools import wraps  # For login_required decorator

# Assuming paths.py defines DATA_DIR and TEMPLATE_DIR correctly relative to project root
try:
    from ahorro.utils.paths import DATA_DIR, TEMPLATE_DIR
except ImportError:
    print(
        "Warning: Could not import from 'ahorro.utils.paths'. Calculating paths manually."
    )
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    TEMPLATE_DIR = SCRIPT_DIR / "templates"


# --- Configuration ---
USERS_FILE = DATA_DIR / "users.csv"
CLIENTS_FILE = DATA_DIR / "clients.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
TICKERS_FILE = DATA_DIR / "tickers.csv"
PRICES_FILE = DATA_DIR / "prices_historical.csv"

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-insecure-secret-key")


# --- Decorator for Login Required ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("You need to be logged in to view this page.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# --- Helper Functions (load_data, load_users, etc. - keep as before) ---
def load_data(file_path, required_columns=None, index_col=None, dtype=None):
    """Loads data from a CSV file using pandas."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            print(f"Info: File not found at {file_path}. Returning empty DataFrame.")
            if required_columns:
                return pd.DataFrame(columns=required_columns)
            else:
                return pd.DataFrame()

        df = pd.read_csv(file_path, index_col=index_col, dtype=dtype)  # Added dtype

        cols_to_check = df.columns
        if index_col and index_col in required_columns:
            pass

        if required_columns:
            if not all(
                col in cols_to_check for col in required_columns if col != index_col
            ):
                print(
                    f"Warning: Missing required columns in {file_path}. Expected: {required_columns}"
                )
                return pd.DataFrame(columns=required_columns)
        return df
    except pd.errors.EmptyDataError:
        print(f"Info: File is empty at {file_path}. Returning empty DataFrame.")
        if required_columns:
            return pd.DataFrame(columns=required_columns)
        else:
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return None


def load_users():
    """Loads user data."""
    return load_data(
        USERS_FILE, required_columns=["user_id", "username", "pwd"], dtype=str
    )


def load_clients():
    """Loads client data."""
    return load_data(
        CLIENTS_FILE,
        required_columns=["client_id", "client_name", "user_id"],
        dtype=str,
    )


def load_transactions():
    """Loads transaction data, ensuring correct types."""
    transaction_columns = ["trx_id", "user_id", "amount", "type", "date", "ticker_id"]
    # Specify dtypes for critical columns
    dtypes = {
        "trx_id": str,
        "user_id": str,
        "amount": float,
        "type": str,
        "date": str,  # Keep as string for now, parse later if needed
        "ticker_id": str,
    }
    df = load_data(
        TRANSACTIONS_FILE, required_columns=transaction_columns, dtype=dtypes
    )
    # Convert amount column after loading if needed, handling potential errors
    if df is not None and "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def load_tickers():
    """Loads ticker data."""
    ticker_columns = ["ticker_id", "ticker_name", "ticker_symbol"]
    return load_data(TICKERS_FILE, required_columns=ticker_columns, dtype=str)


def load_prices():
    """ "Loads price data."""
    price_columns = ["date", "ticker_id", "close_price"]
    return load_data(PRICES_FILE, required_columns=price_columns, dtype=str)


def verify_user(username, password):
    """Verifies username and password hash."""
    users_df = load_users()
    if users_df is None or users_df.empty:
        print("Error or no data: Could not load user data.")
        return None
    try:
        user_record = users_df[
            users_df["username"] == str(username)
        ]  # Ensure string comparison
        if not user_record.empty:
            stored_hash = user_record.iloc[0]["pwd"]
            if isinstance(stored_hash, str) and check_password_hash(
                stored_hash, password
            ):
                user_id_from_df = user_record.iloc[0]["user_id"]
                return int(user_id_from_df)  # Return standard int
            else:
                print(f"Password check failed for user: {username}")
                return None
        else:
            print(f"User not found: {username}")
            return None
    except Exception as e:
        print(f"Error during user verification for {username}: {e}")
        return None


def get_client_name(user_id):
    """Gets the client name associated with a user_id."""
    clients_df = load_clients()
    if clients_df is None or clients_df.empty:
        return "Unknown Client"
    try:
        # Compare user_id as string since we load it as string
        client_record = clients_df[clients_df["user_id"] == str(user_id)]
        if not client_record.empty:
            return client_record.iloc[0]["client_name"]
        else:
            return "Client Not Found"
    except Exception as e:
        print(f"Error getting client name for user_id {user_id}: {e}")
        return "Error Finding Client"


def add_transaction_to_csv(user_id, amount, transaction_type, date_str, ticker_id):
    """Appends a new transaction to the transactions.csv file."""
    transaction_columns = ["trx_id", "user_id", "amount", "type", "date", "ticker_id"]
    new_trx_id = 1
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = TRANSACTIONS_FILE.exists()
        header_needed = not file_exists or os.path.getsize(TRANSACTIONS_FILE) == 0

        if file_exists and not header_needed:
            try:
                df_existing = pd.read_csv(TRANSACTIONS_FILE)
                if not df_existing.empty and "trx_id" in df_existing.columns:
                    valid_ids = pd.to_numeric(
                        df_existing["trx_id"], errors="coerce"
                    ).dropna()
                    if not valid_ids.empty:
                        new_trx_id = int(valid_ids.max()) + 1
                    else:
                        new_trx_id = 1
                else:
                    new_trx_id = 1
            except Exception as e:
                print(
                    f"Error reading existing transactions: {e}. Falling back to ID 1."
                )
                new_trx_id = 1
                header_needed = True

        new_transaction = {
            "trx_id": new_trx_id,
            "user_id": user_id,
            "amount": amount,
            "type": transaction_type,
            "date": date_str,
            "ticker_id": ticker_id,
        }

        with open(TRANSACTIONS_FILE, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=transaction_columns)
            if header_needed:
                writer.writeheader()
            writer.writerow(new_transaction)
        print(f"Successfully added transaction {new_trx_id} for user {user_id}")
        return True
    except Exception as e:
        print(f"Error adding transaction to CSV: {e}")
        return False


def calculate_portfolio(user_id):
    """Calculates current holdings for a given user."""
    transactions_df = load_transactions()
    tickers_df = load_tickers()
    prices_df = load_prices()

    if transactions_df is None or transactions_df.empty:
        return {}  # No transactions, empty portfolio
    if tickers_df is None:
        print("Warning: Could not load tickers data for portfolio calculation.")
        # Decide how to handle: return empty, or try without names?
        # For now, proceed but names/symbols might be missing.
        tickers_df = pd.DataFrame(
            columns=["ticker_id", "ticker_name", "ticker_symbol"]
        )  # Empty df
    if prices_df is None or prices_df.empty:
        print("Warning: Could not load prices data for portfolio calculation.")
        # Proceed without price data, but this might affect calculations
        prices_df = pd.DataFrame(columns=["date", "ticker_id", "close_price"])

    # Ensure user_id comparison is correct type (string based on load_data)
    user_transactions = transactions_df[
        transactions_df["user_id"] == str(user_id)
    ].copy()

    if user_transactions.empty:
        return {}  # No transactions for this user

    # Convert amount to numeric, coercing errors
    user_transactions["amount"] = pd.to_numeric(
        user_transactions["amount"], errors="coerce"
    )
    # Drop rows where amount conversion failed
    user_transactions.dropna(subset=["amount"], inplace=True)

    # Calculate effective amount (+ for buy, - for sell)
    user_transactions["effective_amount"] = user_transactions.apply(
        lambda row: row["amount"] if row["type"].lower() == "buy" else -row["amount"],
        axis=1,
    )

    # Group by ticker_id and sum effective amounts
    holdings = user_transactions.groupby("ticker_id")["effective_amount"].sum()

    # Filter out tickers with zero or negative holdings (optional, depends on requirements)
    holdings = holdings[holdings > 1e-9]  # Use tolerance for float comparison

    if holdings.empty:
        return {}

    # Convert holdings Series to DataFrame
    portfolio_df = holdings.reset_index()
    portfolio_df.rename(columns={"effective_amount": "holding"}, inplace=True)

    # Merge with ticker info to get names and symbols
    # Ensure ticker_id types match for merging (both should be strings here)
    portfolio_df["ticker_id"] = portfolio_df["ticker_id"].astype(str)
    tickers_df["ticker_id"] = tickers_df["ticker_id"].astype(str)

    full_portfolio = pd.merge(portfolio_df, tickers_df, on="ticker_id", how="left")

    # Merge with prices to get latest price
    prices_df["ticker_id"] = prices_df["ticker_id"].astype(str)
    latest_prices = prices_df.sort_values("date").drop_duplicates(
        "ticker_id", keep="last"
    )  # Keep latest price for each ticker
    full_portfolio = pd.merge(
        full_portfolio,
        latest_prices[["ticker_id", "close_price"]],
        on="ticker_id",
        how="left",
    )
    # Rename columns for clarity
    full_portfolio.rename(
        columns={
            "ticker_symbol": "ticker_symbol",
            "ticker_name": "ticker_name",
            "close_price": "worth",
        },
        inplace=True,
    )
    # Calculate total worth
    full_portfolio["worth"] = pd.to_numeric(full_portfolio["worth"], errors="coerce")
    full_portfolio.dropna(subset=["worth"], inplace=True)  # Drop rows with NaN worth
    # Calculate total worth for each holding
    full_portfolio["total_worth"] = (
        full_portfolio["holding"] * full_portfolio["worth"]
    ).round(2)

    # Fill missing ticker names/symbols if merge failed for some IDs
    full_portfolio["ticker_symbol"] = full_portfolio["ticker_symbol"].fillna("UNKNOWN")
    full_portfolio["ticker_name"] = full_portfolio["ticker_name"].fillna(
        "Unknown Ticker"
    )

    # Convert to dictionary format {ticker_id: {details}} for easier template access
    portfolio_summary = full_portfolio.set_index("ticker_id").to_dict("index")

    return portfolio_summary


# --- Flask Routes ---
@app.route("/")
def index():
    """Redirects to home page if logged in, else to login."""
    if "user_id" in session:
        return redirect(url_for("home"))  # Redirect to home now
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles the login process."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("login.html")  # Show login page again

        user_id = verify_user(username, password)
        if user_id is not None:
            client_name = get_client_name(user_id)
            session["user_id"] = user_id
            session["username"] = username
            session["client_name"] = client_name
            flash(f"Login successful!", "success")  # Simpler message
            # Redirect to intended page or home
            next_url = request.args.get("next")
            return redirect(next_url or url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

    # If already logged in, redirect to home
    if "user_id" in session:
        return redirect(url_for("home"))
    # Show login page for GET request
    return render_template("login.html")


# --- NEW Home Route ---
@app.route("/home")
@login_required  # Use the decorator
def home():
    """Displays the home page."""
    # client_name is already in session, base.html displays it
    return render_template("home.html")


# --- NEW Portfolio Route ---
@app.route("/portfolio")
@login_required
def portfolio():
    """Displays the user's portfolio holdings."""
    user_id = session["user_id"]
    portfolio_summary = calculate_portfolio(user_id)
    return render_template("portfolio.html", portfolio_summary=portfolio_summary)


# --- NEW Transactions Route (replaces /welcome) ---
@app.route("/transactions")
@login_required
def transactions_page():
    """Displays the page for adding transactions."""
    today_date = datetime.now().strftime("%Y-%m-%d")
    tickers_df = load_tickers()
    ticker_list = []
    if tickers_df is not None and not tickers_df.empty:
        if all(col in tickers_df.columns for col in ["ticker_id", "ticker_symbol"]):
            # Sort tickers by symbol for dropdown
            tickers_df.sort_values(by="ticker_symbol", inplace=True)
            ticker_list = tickers_df[["ticker_id", "ticker_symbol"]].to_dict("records")
        else:
            print("Warning: 'ticker_id' or 'ticker_symbol' columns missing.")
            flash("Could not load ticker list.", "warning")
    else:
        print("Warning: Failed to load tickers or tickers file is empty.")
        flash("Could not load ticker list.", "warning")

    # Use the new template name
    return render_template(
        "transactions_page.html", today_date=today_date, tickers=ticker_list
    )


# --- add_transaction route (mostly unchanged, redirects to transactions_page) ---
@app.route("/add_transaction", methods=["POST"])
@login_required  # Add decorator here too
def add_transaction():
    """Handles the form submission for adding a new transaction."""
    user_id = session["user_id"]
    transaction_type = request.form.get("type")
    amount_str = request.form.get("amount")
    date_str = request.form.get("date")
    selected_ticker_symbol = request.form.get("ticker_symbol")

    # --- Basic Input Validation ---
    if not all([transaction_type, amount_str, date_str, selected_ticker_symbol]):
        flash("All transaction fields are required.", "danger")
        return redirect(
            url_for("transactions_page")
        )  # Redirect back to transactions page
    # ... (keep other validation as before) ...
    if transaction_type not in ["buy", "sell"]:
        flash("Invalid transaction type selected.", "danger")
        return redirect(url_for("transactions_page"))
    try:
        amount = float(amount_str)
        if amount <= 0:
            flash("Amount must be a positive number.", "danger")
            return redirect(url_for("transactions_page"))
    except ValueError:
        flash("Invalid amount entered. Please enter a number.", "danger")
        return redirect(url_for("transactions_page"))
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        flash("Invalid date format. Please use YYYY-MM-DD.", "danger")
        return redirect(url_for("transactions_page"))

    # --- Look up ticker_id ---
    tickers_df = load_tickers()
    ticker_id = None
    if (
        tickers_df is not None
        and not tickers_df.empty
        and "ticker_symbol" in tickers_df.columns
        and "ticker_id" in tickers_df.columns
    ):
        ticker_record = tickers_df[
            tickers_df["ticker_symbol"].astype(str) == str(selected_ticker_symbol)
        ]
        if not ticker_record.empty:
            ticker_id = ticker_record.iloc[0]["ticker_id"]
        else:
            flash(
                f"Selected ticker symbol '{selected_ticker_symbol}' not found.",
                "danger",
            )
            return redirect(url_for("transactions_page"))
    else:
        flash("Failed to load ticker data for validation.", "danger")
        return redirect(url_for("transactions_page"))

    # --- Add transaction ---
    if ticker_id is not None:
        if add_transaction_to_csv(
            user_id, amount, transaction_type, date_str, ticker_id
        ):
            flash("Transaction added successfully!", "success")
        else:
            flash("Failed to add transaction.", "danger")
    else:
        flash("Could not determine Ticker ID.", "danger")

    return redirect(url_for("transactions_page"))  # Redirect back


@app.route("/logout")
def logout():
    """Logs the user out."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# --- Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ["1", "true"]
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
