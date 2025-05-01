import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import sys
import time  # Import time for potential delays between API calls

from ahorro.utils.paths import DATA_DIR

# --- Configuration ---
# Input file with ticker symbols and IDs
TICKERS_INPUT_FILE = DATA_DIR / "tickers.csv"
# Output file for fetched historical prices
PRICES_OUTPUT_FILE = DATA_DIR / "prices_historical.csv"

# Columns expected in tickers.csv
TICKER_COLUMNS = ["ticker_id", "ticker_name", "ticker_symbol"]
# Columns for the output prices_historical.csv
PRICE_COLUMNS = [
    "date",
    "ticker_id",
    "close_price",
]

# --- Helper Functions ---


def load_ticker_mapping(file_path):
    """
    Loads ticker symbols and their corresponding IDs from the specified CSV file.
    Expects columns 'ticker_id' and 'ticker_symbol'.
    Returns a dictionary mapping symbol -> id, or None on error.
    """
    print(f"Attempting to load ticker mapping from: {file_path}")
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            print(f"Error: Tickers file not found at '{file_path}'.")
            return None

        # Read required columns, ensuring they are treated as strings initially
        tickers_df = pd.read_csv(
            file_path, usecols=["ticker_id", "ticker_symbol"], dtype=str
        )

        # Check if required columns were actually loaded (usecols might fail silently if columns missing)
        if (
            "ticker_symbol" not in tickers_df.columns
            or "ticker_id" not in tickers_df.columns
        ):
            print(
                f"Error: Columns 'ticker_id' and/or 'ticker_symbol' not found in '{file_path}'."
            )
            return None

        if tickers_df.empty:
            print(f"Warning: Tickers file '{file_path}' is empty.")
            return {}

        # Remove rows where either symbol or id is missing
        tickers_df.dropna(subset=["ticker_id", "ticker_symbol"], inplace=True)
        # Remove duplicates based on symbol, keeping the first occurrence
        tickers_df.drop_duplicates(subset=["ticker_symbol"], keep="first", inplace=True)

        if tickers_df.empty:
            print(f"Warning: No valid ticker symbol/ID pairs found after cleaning.")
            return {}

        # Create the dictionary mapping: symbol -> id
        # Set index to symbol for easy lookup
        ticker_map = tickers_df.set_index("ticker_symbol")["ticker_id"].to_dict()

        print(f"Found {len(ticker_map)} unique ticker symbol-ID pairs.")
        return ticker_map

    except pd.errors.EmptyDataError:
        print(f"Warning: Tickers file '{file_path}' is empty.")
        return {}
    except ValueError as ve:
        # Specifically catch errors if usecols fails because columns aren't present
        print(
            f"Error: Missing required columns ('ticker_id', 'ticker_symbol') in '{file_path}'. Details: {ve}"
        )
        return None
    except Exception as e:
        print(f"Error reading tickers CSV file '{file_path}': {e}")
        return None


# --- Updated function to fetch HISTORICAL prices using the mapping ---
def fetch_historical_prices(ticker_mapping):
    """
    Fetches the complete historical closing prices using ticker symbols from the mapping.
    Associates the fetched data with the corresponding ticker_id from the mapping.
    Returns a list of dictionaries, each representing one closing price for one day for one ticker_id.
    """
    if not ticker_mapping:
        print("No ticker symbol-ID mapping provided.")
        return []

    symbols = list(ticker_mapping.keys())  # Get symbols to fetch
    print(f"Fetching historical prices for {len(symbols)} symbols...")
    all_price_data = []  # List to hold all historical data points
    fetch_errors = []

    for symbol in symbols:
        if not symbol or pd.isna(symbol):
            print("Skipping empty or invalid symbol.")
            continue

        ticker_id = ticker_mapping.get(symbol)  # Get the ID for this symbol
        if ticker_id is None:  # Should not happen if mapping is generated correctly
            print(
                f"    Internal Error: Could not find ticker_id for symbol '{symbol}' in mapping. Skipping."
            )
            fetch_errors.append(symbol + " (ID missing)")
            continue

        print(f"  Fetching historical data for: {symbol} (ID: {ticker_id})...")
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="max", interval="1d")

            if hist.empty:
                print(f"    Warning: No history found for symbol '{symbol}'. Skipping.")
                fetch_errors.append(symbol)
                continue

            hist = hist.reset_index()

            if "Date" not in hist.columns or "Close" not in hist.columns:
                print(
                    f"    Warning: Required columns ('Date', 'Close') not found in history for '{symbol}'. Skipping."
                )
                fetch_errors.append(symbol)
                continue

            symbol_hist = hist[["Date", "Close"]].copy()
            # --- Add the ticker_id instead of ticker_symbol ---
            symbol_hist["ticker_id"] = ticker_id
            # --- Rename columns ---
            symbol_hist.rename(
                columns={"Date": "date", "Close": "close_price"}, inplace=True
            )

            # Format date column
            if pd.api.types.is_datetime64tz_dtype(symbol_hist["date"]):
                symbol_hist["date"] = symbol_hist["date"].dt.strftime("%Y-%m-%d")
            else:
                try:
                    symbol_hist["date"] = pd.to_datetime(
                        symbol_hist["date"]
                    ).dt.strftime("%Y-%m-%d")
                except Exception as date_err:
                    print(
                        f"    Warning: Could not format date for {symbol}: {date_err}. Skipping date conversion."
                    )

            # Append the processed data (now with ticker_id)
            all_price_data.extend(symbol_hist.to_dict("records"))
            print(
                f"    -> Processed {len(symbol_hist)} historical records for ID {ticker_id}."
            )

            time.sleep(0.1)

        except Exception as e:
            print(
                f"    Error fetching/processing data for symbol '{symbol}' (ID: {ticker_id}): {e}"
            )
            fetch_errors.append(symbol)
            continue

    if fetch_errors:
        print(
            "\nWarning: Could not fetch or process data for the following symbols/issues:"
        )
        for err_symbol in fetch_errors:
            print(f"  - {err_symbol}")

    print(f"\nTotal historical records collected: {len(all_price_data)}")
    return all_price_data


# --- Updated function to save HISTORICAL prices (columns changed) ---
def save_prices_to_csv(price_data, output_file):
    """
    Saves the fetched historical price data (with ticker_id) to a CSV file,
    overwriting the existing file.
    """
    if not price_data:
        print("No historical price data to save.")
        return False

    print(
        f"\nAttempting to save {len(price_data)} historical price records to: {output_file}"
    )
    try:
        # Convert list of dictionaries to DataFrame using the updated PRICE_COLUMNS
        prices_df = pd.DataFrame(
            price_data, columns=PRICE_COLUMNS
        )  # Uses ['date', 'ticker_id', 'close_price']

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        prices_df.to_csv(output_file, index=False, float_format="%.4f")
        print(f"Successfully saved historical price data to '{output_file}'.")
        return True

    except Exception as e:
        print(f"Error writing historical prices CSV file '{output_file}': {e}")
        return False


# --- Main Execution ---
if __name__ == "__main__":
    print("-" * 30)
    print("Starting Historical Ticker Price Update Script (using Ticker ID)")
    print(f"Time: {datetime.now()}")
    print("-" * 30)

    # 1. Load ticker symbol-ID mapping
    ticker_map = load_ticker_mapping(TICKERS_INPUT_FILE)

    if ticker_map is None:
        print("\nExiting script due to error loading ticker mapping.")
        sys.exit(1)
    elif not ticker_map:
        print("\nNo ticker symbol-ID mapping found. Exiting script.")
        sys.exit(0)

    # 2. Fetch HISTORICAL prices using the mapping
    fetched_data = fetch_historical_prices(ticker_map)

    if not fetched_data:
        print("\nNo historical price data was successfully fetched. Exiting script.")
        sys.exit(0)

    # 3. Save fetched prices (using the updated function)
    success = save_prices_to_csv(fetched_data, PRICES_OUTPUT_FILE)

    print("-" * 30)
    if success:
        print("Historical price update script finished successfully.")
        sys.exit(0)
    else:
        print("Historical price update script finished with errors during saving.")
        sys.exit(1)
