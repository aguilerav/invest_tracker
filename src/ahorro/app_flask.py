import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash
from pathlib import Path
from datetime import datetime
import csv  # Import csv module for more control over writing

# Assuming paths.py defines DATA_DIR and TEMPLATE_DIR correctly relative to project root
try:
    # Use absolute import based on the installed package 'ahorro'
    from ahorro.utils.paths import DATA_DIR, TEMPLATE_DIR
except ImportError:
    # Fallback for environments where the package might not be installed in editable mode yet
    print(
        "Warning: Could not import from 'ahorro.utils.paths'. Calculating paths manually."
    )
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = (
        SCRIPT_DIR.parent.parent
    )  # Navigate up from src/ahorro to project root
    DATA_DIR = PROJECT_ROOT / "data"
    TEMPLATE_DIR = (
        SCRIPT_DIR / "templates"
    )  # Assuming templates are inside src/ahorro/templates


# --- Configuration ---
USERS_FILE = DATA_DIR / "users.csv"
CLIENTS_FILE = DATA_DIR / "clients.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
TICKERS_FILE = DATA_DIR / "tickers.csv"  # <-- Path to tickers file

# --- Flask App Initialization ---
# Ensure TEMPLATE_DIR is a string for Flask
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Secret key is needed for session management and flash messages
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-insecure-secret-key")


# --- Helper Functions ---
def load_data(file_path, required_columns=None, index_col=None):
    """Loads data from a CSV file using pandas."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            print(f"Info: File not found at {file_path}. Returning empty DataFrame.")
            if required_columns:
                return pd.DataFrame(columns=required_columns)
            else:
                return pd.DataFrame()

        # Read CSV, potentially setting an index column if specified
        df = pd.read_csv(file_path, index_col=index_col)

        # Check required columns (adjust check if index_col is used)
        cols_to_check = df.columns
        if index_col and index_col in required_columns:
            # If index_col is required, check it separately or adjust required_columns list
            pass  # Assuming index is handled correctly if specified

        if required_columns:
            if not all(
                col in cols_to_check for col in required_columns if col != index_col
            ):
                print(
                    f"Warning: Missing required columns in {file_path}. Expected: {required_columns}"
                )
                return pd.DataFrame(
                    columns=required_columns
                )  # Return empty df matching expected structure
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
    return load_data(USERS_FILE, required_columns=["user_id", "username", "pwd"])


def load_clients():
    """Loads client data."""
    return load_data(
        CLIENTS_FILE, required_columns=["client_id", "client_name", "user_id"]
    )


def load_transactions():
    """Loads transaction data."""
    transaction_columns = ["trx_id", "user_id", "amount", "type", "date", "ticker_id"]
    return load_data(TRANSACTIONS_FILE, required_columns=transaction_columns)


# --- NEW: Load Tickers Function ---
def load_tickers():
    """
    Loads ticker data from tickers.csv.
    Returns a DataFrame or None on error.
    Expected columns: ticker_id, ticker_name, ticker_symbol
    """
    ticker_columns = ["ticker_id", "ticker_name", "ticker_symbol"]
    # Consider setting ticker_symbol as index for faster lookups if needed often
    # return load_data(TICKERS_FILE, required_columns=ticker_columns, index_col='ticker_symbol')
    return load_data(TICKERS_FILE, required_columns=ticker_columns)


# --- End NEW ---


def verify_user(username, password):
    """Verifies username and password hash."""
    users_df = load_users()
    if users_df is None or users_df.empty:
        print("Error or no data: Could not load user data.")
        return None
    try:
        user_record = users_df[users_df["username"].astype(str) == str(username)]
        if not user_record.empty:
            stored_hash = user_record.iloc[0]["pwd"]
            if isinstance(stored_hash, str) and check_password_hash(
                stored_hash, password
            ):
                user_id_from_df = user_record.iloc[0]["user_id"]
                return int(user_id_from_df)
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
        # Ensure comparison uses compatible types. If user_id in df is int, compare as int.
        clients_df["user_id"] = pd.to_numeric(clients_df["user_id"], errors="coerce")
        client_record = clients_df[clients_df["user_id"] == int(user_id)]
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
                # More robust way to get max ID using pandas if file isn't huge
                df_existing = pd.read_csv(TRANSACTIONS_FILE)
                if not df_existing.empty and "trx_id" in df_existing.columns:
                    # Ensure trx_id is numeric before finding max
                    valid_ids = pd.to_numeric(
                        df_existing["trx_id"], errors="coerce"
                    ).dropna()
                    if not valid_ids.empty:
                        new_trx_id = int(valid_ids.max()) + 1
                    else:
                        new_trx_id = 1  # No valid numeric IDs found
                else:
                    new_trx_id = 1  # File exists but empty or no trx_id column
            except Exception as e:
                print(
                    f"Error reading existing transactions with pandas: {e}. Falling back to ID 1."
                )
                new_trx_id = 1  # Fallback ID
                header_needed = True  # Assume header might be corrupted

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


# --- Flask Routes ---
@app.route("/")
def index():
    """Redirects to login or welcome page."""
    if "user_id" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles the login process."""
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
            flash(f"Login successful! Welcome {client_name}.", "success")
            return redirect(url_for("welcome"))
        else:
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

    if "user_id" in session:
        return redirect(url_for("welcome"))
    return render_template("login.html")


@app.route("/welcome")
def welcome():
    """Displays the welcome page with transaction form."""
    if "user_id" not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for("login"))

    client_name = session.get("client_name", "Guest")
    today_date = datetime.now().strftime("%Y-%m-%d")

    # --- Load Tickers for Dropdown ---
    tickers_df = load_tickers()
    ticker_list = []
    if tickers_df is not None and not tickers_df.empty:
        # Create a list of dictionaries for the template
        # Ensure columns exist before accessing
        if all(col in tickers_df.columns for col in ["ticker_id", "ticker_symbol"]):
            ticker_list = tickers_df[["ticker_id", "ticker_symbol"]].to_dict("records")
        else:
            print(
                "Warning: 'ticker_id' or 'ticker_symbol' columns missing in tickers.csv"
            )
            flash("Could not load ticker list.", "warning")
    else:
        print("Warning: Failed to load tickers or tickers file is empty.")
        flash("Could not load ticker list.", "warning")
    # --- End Load Tickers ---

    return render_template(
        "welcome.html",
        client_name=client_name,
        today_date=today_date,
        tickers=ticker_list,
    )  # Pass tickers to template


# --- UPDATED ROUTE for adding transactions ---
@app.route("/add_transaction", methods=["POST"])
def add_transaction():
    """Handles the form submission for adding a new transaction."""
    if "user_id" not in session:
        flash("Authentication required to add transaction.", "danger")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    transaction_type = request.form.get("type")
    amount_str = request.form.get("amount")
    date_str = request.form.get("date")
    # --- Get selected ticker SYMBOL from form ---
    selected_ticker_symbol = request.form.get("ticker_symbol")

    # --- Basic Input Validation ---
    if not all([transaction_type, amount_str, date_str, selected_ticker_symbol]):
        flash("All transaction fields are required.", "danger")
        return redirect(
            url_for("welcome")
        )  # Redirect back to welcome to show form again
    if transaction_type not in ["buy", "sell"]:
        flash("Invalid transaction type selected.", "danger")
        return redirect(url_for("welcome"))
    try:
        amount = float(amount_str)
        if amount <= 0:
            flash("Amount must be a positive number.", "danger")
            return redirect(url_for("welcome"))
    except ValueError:
        flash("Invalid amount entered. Please enter a number.", "danger")
        return redirect(url_for("welcome"))
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        flash("Invalid date format. Please use YYYY-MM-DD.", "danger")
        return redirect(url_for("welcome"))
    # --- End Validation ---

    # --- Look up ticker_id based on selected_ticker_symbol ---
    tickers_df = load_tickers()
    ticker_id = None
    if (
        tickers_df is not None
        and not tickers_df.empty
        and "ticker_symbol" in tickers_df.columns
        and "ticker_id" in tickers_df.columns
    ):
        # Ensure comparison is string vs string
        ticker_record = tickers_df[
            tickers_df["ticker_symbol"].astype(str) == str(selected_ticker_symbol)
        ]
        if not ticker_record.empty:
            # Get the first matching ticker_id
            ticker_id = ticker_record.iloc[0]["ticker_id"]
            # Optional: Convert ticker_id to appropriate type if needed (e.g., int)
            # try:
            #     ticker_id = int(ticker_id)
            # except ValueError:
            #     flash(f"Invalid ticker ID format found for symbol {selected_ticker_symbol}.", "danger")
            #     return redirect(url_for("welcome"))
        else:
            flash(
                f"Selected ticker symbol '{selected_ticker_symbol}' not found.",
                "danger",
            )
            return redirect(url_for("welcome"))
    else:
        flash("Failed to load ticker data for validation.", "danger")
        return redirect(url_for("welcome"))
    # --- End Ticker ID Lookup ---

    # Attempt to add the transaction to the CSV using the found ticker_id
    # Ensure ticker_id is not None before proceeding
    if ticker_id is not None:
        if add_transaction_to_csv(
            user_id, amount, transaction_type, date_str, ticker_id
        ):
            flash("Transaction added successfully!", "success")
        else:
            flash(
                "Failed to add transaction. Please check logs or try again.", "danger"
            )
    else:
        # This case should ideally be caught earlier, but as a safeguard
        flash("Could not determine Ticker ID for the selected symbol.", "danger")

    return redirect(url_for("welcome"))


# --- End UPDATED ROUTE ---


@app.route("/logout")
def logout():
    """Logs the user out."""
    session.clear()  # Clear all session data
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# --- Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ["1", "true"]
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
