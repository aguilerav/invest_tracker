import os
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
)
from werkzeug.security import check_password_hash
from pathlib import Path
from datetime import datetime
import csv
from functools import wraps

# --- Imports and Path Setup (Keep as before) ---
try:
    from ahorro.utils.paths import DATA_DIR, TEMPLATE_DIR, STATIC_DIR
except ImportError:
    print(
        "Warning: Could not import from 'ahorro.utils.paths'. Calculating paths manually."
    )
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    TEMPLATE_DIR = PROJECT_ROOT / "templates"
    STATIC_DIR = PROJECT_ROOT / "static"

# --- Configuration ---
USERS_FILE = DATA_DIR / "users.csv"
CLIENTS_FILE = DATA_DIR / "clients.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
TICKERS_FILE = DATA_DIR / "tickers.csv"
PRICES_FILE = DATA_DIR / "prices_historical.csv"

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=STATIC_DIR)
# IMPORTANT: Ensure your static folder is correctly detected.
# By default, Flask looks for a 'static' folder sibling to the template folder
# or specified via static_folder=...
# If your structure is src/ahorro/app_flask.py, src/ahorro/templates/, src/ahorro/static/,
# it should work automatically.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-insecure-secret-key")


# --- Decorator for Login Required (Keep as before) ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("You need to be logged in to view this page.", "warning")
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# --- Helper Functions (load_data, load_users, etc.) ---
# ... (load_data, load_users, load_clients, load_transactions, load_tickers remain the same) ...
def load_data(
    file_path, required_columns=None, index_col=None, dtype=None, parse_dates=None
):
    """Loads data from a CSV file using pandas."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            if required_columns:
                return pd.DataFrame(columns=required_columns)
            else:
                return pd.DataFrame()
        df = pd.read_csv(
            file_path, index_col=index_col, dtype=dtype, parse_dates=parse_dates
        )
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
    return load_data(
        USERS_FILE, required_columns=["user_id", "username", "pwd"], dtype=str
    )


def load_clients():
    return load_data(
        CLIENTS_FILE,
        required_columns=["client_id", "client_name", "user_id"],
        dtype=str,
    )


def load_transactions():
    transaction_columns = ["trx_id", "user_id", "amount", "type", "date", "ticker_id"]
    dtypes = {
        "trx_id": str,
        "user_id": str,
        "amount": float,
        "type": str,
        "date": str,
        "ticker_id": str,
    }
    df = load_data(
        TRANSACTIONS_FILE, required_columns=transaction_columns, dtype=dtypes
    )
    if df is not None and "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def load_tickers():
    ticker_columns = ["ticker_id", "ticker_name", "ticker_symbol"]
    return load_data(TICKERS_FILE, required_columns=ticker_columns, dtype=str)


def load_latest_prices():
    """Loads latest price per ticker_id from historical data."""
    price_columns = ["date", "ticker_id", "close_price"]
    prices_df = load_data(
        PRICES_FILE, required_columns=price_columns, dtype={"ticker_id": str}
    )
    if prices_df is None or prices_df.empty:
        return pd.DataFrame(columns=["ticker_id", "price", "price_date"])
    prices_df["date_dt"] = pd.to_datetime(prices_df["date"], errors="coerce")
    prices_df["close_price"] = pd.to_numeric(prices_df["close_price"], errors="coerce")
    prices_df.dropna(subset=["date_dt", "ticker_id", "close_price"], inplace=True)
    if prices_df.empty:
        return pd.DataFrame(columns=["ticker_id", "price", "price_date"])
    latest_idx = prices_df.loc[prices_df.groupby("ticker_id")["date_dt"].idxmax()]
    latest_prices = latest_idx[["ticker_id", "close_price", "date_dt"]].copy()
    latest_prices.rename(
        columns={"close_price": "price", "date_dt": "price_date"}, inplace=True
    )
    latest_prices["price_date"] = latest_prices["price_date"].dt.strftime("%Y-%m-%d")
    latest_prices["ticker_id"] = latest_prices["ticker_id"].astype(str)
    return latest_prices[["ticker_id", "price", "price_date"]]


# --- verify_user, get_client_name, add_transaction_to_csv (Keep as before) ---
def verify_user(username, password):
    users_df = load_users()
    if users_df is None or users_df.empty:
        return None
    try:
        user_record = users_df[users_df["username"] == str(username)]
        if not user_record.empty:
            stored_hash = user_record.iloc[0]["pwd"]
            if isinstance(stored_hash, str) and check_password_hash(
                stored_hash, password
            ):
                return int(user_record.iloc[0]["user_id"])
        return None
    except Exception as e:
        print(f"Error during user verification: {e}")
        return None


def get_client_name(user_id):
    clients_df = load_clients()
    if clients_df is None or clients_df.empty:
        return "Unknown Client"
    try:
        client_record = clients_df[clients_df["user_id"] == str(user_id)]
        if not client_record.empty:
            return client_record.iloc[0]["client_name"]
        else:
            return "Client Not Found"
    except Exception as e:
        print(f"Error getting client name: {e}")
        return "Error Finding Client"


def add_transaction_to_csv(
    user_id, amount, price, transaction_type, date_str, ticker_id
):
    transaction_columns = [
        "trx_id",
        "user_id",
        "amount",
        "price",
        "type",
        "date",
        "ticker_id",
    ]
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
                print(f"Error reading existing transactions: {e}. Fallback ID 1.")
                new_trx_id = 1
                header_needed = True
        new_transaction = {
            "trx_id": new_trx_id,
            "user_id": user_id,
            "amount": amount,
            "price": price,
            "type": transaction_type,
            "date": date_str,
            "ticker_id": ticker_id,
        }
        with open(TRANSACTIONS_FILE, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=transaction_columns)
            if header_needed:
                writer.writeheader()
            writer.writerow(new_transaction)
        return True
    except Exception as e:
        print(f"Error adding transaction: {e}")
        return False


def calculate_portfolio(user_id):
    """
    Calculates current holdings and their values for a given user.
    Includes path to ticker icon.
    Returns:
        - portfolio_summary (dict): Details of each holding {ticker_id: {details...}}
        - overall_total_worth (float): Sum of the current value of all holdings.
    """
    transactions_df = load_transactions()
    tickers_df = load_tickers()
    latest_prices_df = load_latest_prices()

    portfolio_summary = {}
    overall_total_worth = 0.0

    if transactions_df is None:
        return portfolio_summary, overall_total_worth

    user_transactions = transactions_df[
        transactions_df["user_id"] == str(user_id)
    ].copy()
    if user_transactions.empty:
        return portfolio_summary, overall_total_worth

    user_transactions["amount"] = pd.to_numeric(
        user_transactions["amount"], errors="coerce"
    )
    user_transactions.dropna(subset=["amount"], inplace=True)
    user_transactions["effective_amount"] = user_transactions.apply(
        lambda row: row["amount"] if row["type"].lower() == "buy" else -row["amount"],
        axis=1,
    )
    holdings = user_transactions.groupby("ticker_id")["effective_amount"].sum()
    holdings = holdings[holdings > 1e-9]

    if holdings.empty:
        return portfolio_summary, overall_total_worth

    portfolio_df = holdings.reset_index().copy()
    portfolio_df.rename(columns={"effective_amount": "holding"}, inplace=True)
    portfolio_df["ticker_id"] = portfolio_df["ticker_id"].astype(str)

    # Merge with tickers
    if tickers_df is not None:
        tickers_df["ticker_id"] = tickers_df["ticker_id"].astype(str)
        portfolio_df = pd.merge(portfolio_df, tickers_df, on="ticker_id", how="left")
    else:
        portfolio_df["ticker_symbol"] = "UNKNOWN"
        portfolio_df["ticker_name"] = "Unknown Ticker"

    # Merge with prices
    if latest_prices_df is not None and not latest_prices_df.empty:
        latest_prices_df["ticker_id"] = latest_prices_df["ticker_id"].astype(str)
        portfolio_df = pd.merge(
            portfolio_df,
            latest_prices_df[["ticker_id", "price", "price_date"]],
            on="ticker_id",
            how="left",
        )
        portfolio_df["price"] = pd.to_numeric(portfolio_df["price"], errors="coerce")
    else:
        portfolio_df["price"] = None
        portfolio_df["price_date"] = None

    # Fill missing info
    portfolio_df["ticker_symbol"] = portfolio_df["ticker_symbol"].fillna("UNKNOWN")
    portfolio_df["ticker_name"] = portfolio_df["ticker_name"].fillna("Unknown Ticker")
    portfolio_df["price"] = portfolio_df["price"].fillna(0.0)
    portfolio_df["price_date"] = portfolio_df["price_date"].fillna("N/A")

    # Calculate value for each holding
    portfolio_df["total_worth"] = portfolio_df["holding"] * portfolio_df["price"]

    # Calculate overall total worth
    overall_total_worth = portfolio_df["total_worth"].sum()

    # Convert to dictionary
    portfolio_summary = portfolio_df.set_index("ticker_id").to_dict("index")

    return portfolio_summary, overall_total_worth


def get_holding_for_ticker(user_id, ticker_id):
    """Calculates the current holding amount for a specific user and ticker."""
    transactions_df = load_transactions()
    if transactions_df is None or transactions_df.empty:
        return 0.0  # No transactions, so holding is 0

    # Filter for the specific user and ticker
    user_ticker_transactions = transactions_df[
        (transactions_df["user_id"] == str(user_id))
        & (transactions_df["ticker_id"] == str(ticker_id))
    ].copy()

    if user_ticker_transactions.empty:
        return 0.0  # No transactions for this specific ticker

    # Calculate effective amount (buy adds, sell subtracts)
    user_ticker_transactions["amount"] = pd.to_numeric(
        user_ticker_transactions["amount"], errors="coerce"
    )
    user_ticker_transactions.dropna(
        subset=["amount"], inplace=True
    )  # Drop rows where amount couldn't be converted

    user_ticker_transactions["effective_amount"] = user_ticker_transactions.apply(
        lambda row: (
            row["amount"] if str(row["type"]).lower() == "buy" else -row["amount"]
        ),
        axis=1,
    )

    # Sum up to get the current holding
    current_holding = user_ticker_transactions["effective_amount"].sum()

    # Ensure holding isn't negative due to potential data issues, round slightly to avoid floating point dust
    return max(
        0.0, round(current_holding, 10)
    )  # Round to 8 decimal places for precision


# --- Flask Routes ---
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("login.html")
        user_id = verify_user(username, password)
        if user_id is not None:
            client_name = get_client_name(user_id)
            session["user_id"] = user_id
            session["username"] = username
            session["client_name"] = client_name
            flash(f"Login successful!", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("home"))
        else:
            flash("Invalid username or password.", "danger")
            return render_template("login.html")
    if "user_id" in session:
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/portfolio")
@login_required
def portfolio():
    user_id = session["user_id"]
    portfolio_summary, overall_total_worth = calculate_portfolio(user_id)
    return render_template(
        "portfolio.html",
        portfolio_summary=portfolio_summary,
        overall_total_worth=overall_total_worth,
    )


@app.route("/transactions")
@login_required
def transactions_page():
    today_date = datetime.now().strftime("%Y-%m-%d")
    tickers_df = load_tickers()
    ticker_list = []
    if tickers_df is not None and not tickers_df.empty:
        if all(col in tickers_df.columns for col in ["ticker_id", "ticker_symbol"]):
            tickers_df.sort_values(by="ticker_symbol", inplace=True)
            ticker_list = tickers_df[["ticker_id", "ticker_symbol"]].to_dict("records")
        else:
            flash("Could not load ticker list (missing columns).", "warning")
    else:
        flash("Could not load ticker list (file error or empty).", "warning")
    return render_template(
        "transactions_page.html", today_date=today_date, tickers=ticker_list
    )


@app.route("/get_holding_amount/<ticker_symbol>")
@login_required
def get_holding_amount_route(ticker_symbol):
    """API endpoint to get the current holding amount for a ticker symbol."""
    user_id = session.get("user_id")
    if not user_id:
        # This shouldn't happen due to @login_required, but good practice
        return jsonify({"error": "User not logged in"}), 401

    if not ticker_symbol:
        return jsonify({"error": "Ticker symbol required"}), 400

    # Find the ticker_id for the given symbol
    tickers_df = load_tickers()
    ticker_id = None
    if (
        tickers_df is not None
        and not tickers_df.empty
        and "ticker_symbol" in tickers_df.columns
        and "ticker_id" in tickers_df.columns
    ):
        ticker_record = tickers_df[
            tickers_df["ticker_symbol"].astype(str) == str(ticker_symbol)
        ]
        if not ticker_record.empty:
            ticker_id = ticker_record.iloc[0]["ticker_id"]
        else:
            return jsonify({"error": f"Ticker symbol '{ticker_symbol}' not found"}), 404
    else:
        return jsonify({"error": "Ticker data unavailable"}), 500

    # Calculate the holding using the helper function
    holding_amount = get_holding_for_ticker(user_id, ticker_id)

    return jsonify({"holding": holding_amount})


@app.route("/add_transaction", methods=["POST"])
@login_required
def add_transaction():
    user_id = session["user_id"]
    # Get form data
    transaction_type = request.form.get("type")
    amount_str = request.form.get("amount")
    price_str = request.form.get("price")
    date_str = request.form.get("date")
    selected_ticker_symbol = request.form.get("ticker_symbol")
    # Validation
    print(
        f"Received transaction data: {transaction_type}, {amount_str}, {price_str}, {date_str}, {selected_ticker_symbol}"
    )
    if not all(
        [transaction_type, amount_str, price_str, date_str, selected_ticker_symbol]
    ):
        flash("All transaction fields are required.", "danger")
        return redirect(url_for("transactions_page"))
    if transaction_type not in ["buy", "sell"]:
        flash("Invalid transaction type.", "danger")
        return redirect(url_for("transactions_page"))
    try:
        amount = float(amount_str)
        assert amount > 0
    except:
        flash("Amount must be a positive number.", "danger")
        return redirect(url_for("transactions_page"))
    try:
        price = float(price_str)
        assert price > 0
    except:
        flash("Price must be a positive number.", "danger")
        return redirect(url_for("transactions_page"))
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except:
        flash("Invalid date format.", "danger")
        return redirect(url_for("transactions_page"))
    # Look up ticker_id
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
            flash(f"Ticker symbol '{selected_ticker_symbol}' not found.", "danger")
            return redirect(url_for("transactions_page"))
    else:
        flash("Ticker data unavailable for validation.", "danger")
        return redirect(url_for("transactions_page"))
    # Add transaction
    if ticker_id is not None:
        if add_transaction_to_csv(
            user_id, amount, price, transaction_type, date_str, ticker_id
        ):
            flash("Transaction added successfully!", "success")
        else:
            flash("Failed to add transaction.", "danger")
    else:
        flash("Could not determine Ticker ID.", "danger")
    return redirect(url_for("transactions_page"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# --- Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ["1", "true"]
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
