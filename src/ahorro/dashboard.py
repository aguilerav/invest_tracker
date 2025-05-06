import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# --- Dash Imports ---
import dash
from dash import dcc, html, Input, Output, State

# Assuming ahorro.utils.paths provides DATA_DIR correctly
try:
    from ahorro.utils.paths import DATA_DIR
except ImportError:
    print("Warning: Could not import DATA_DIR. Using relative path assumption.")
    # Fallback if the import doesn't work - adjust as necessary
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_DIR = SCRIPT_DIR.parent / "data"  # Common structure 1
    if not DATA_DIR.exists():
        # Try another common structure if the first fails
        DATA_DIR = SCRIPT_DIR.parent.parent / "data"  # Common structure 2


PRICES_FILE = DATA_DIR / "prices_historical.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"


# (Keep load_data, load_transactions, load_prices, calculate_portfolio_history functions as before)
# ... (Paste those functions here) ...
def load_data(
    file_path, required_columns=None, index_col=None, dtype=None, parse_dates=None
):
    """Loads data from a CSV file using pandas with enhanced logging."""
    func_name = "load_data"  # For logging context
    try:
        # Ensure DATA_DIR exists before trying to read from it
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            # print(f"[{func_name}] Warning: File not found {file_path}. Returning empty DF.") # Reduce noise
            return (
                pd.DataFrame(columns=required_columns)
                if required_columns
                else pd.DataFrame()
            )

        df = pd.read_csv(
            file_path, index_col=index_col, dtype=dtype, parse_dates=parse_dates
        )

        if df.empty:
            # Check if columns exist even if empty (read_csv might return columns)
            if required_columns and not all(
                col in df.columns for col in required_columns if col != index_col
            ):
                # print(f"[{func_name}] Warning: File {file_path} is empty and missing required columns. Returning empty DataFrame with required columns.") # Reduce noise
                return pd.DataFrame(columns=required_columns)
            elif df.empty:
                # print(f"[{func_name}] Warning: File {file_path} is empty but columns exist or none required. Returning empty DataFrame.") # Reduce noise
                return df

        # Check for required columns after loading data
        cols_to_check = list(df.columns)
        missing_cols = []
        if required_columns:
            for col in required_columns:
                # Check if column exists, ignoring index if it was specified
                if col != index_col and col not in cols_to_check:
                    missing_cols.append(col)

            if missing_cols:
                # Log clearly which columns are missing
                # print(
                #     f"[{func_name}] Warning: Missing required columns in {file_path}. "
                #     f"Missing: {missing_cols}. Expected: {required_columns}. "
                #     f"Returning DataFrame with missing columns added as NA."
                # ) # Reduce noise
                # Add missing columns as NA
                for col in missing_cols:
                    # Determine appropriate NA type based on potential dtype if available
                    na_val = pd.NA
                    if dtype and col in dtype:
                        if np.issubdtype(dtype[col], np.number):
                            na_val = np.nan
                        elif np.issubdtype(dtype[col], np.datetime64):
                            na_val = pd.NaT
                    df[col] = na_val
                # Ensure the final DataFrame has at least the required columns
                final_cols = list(df.columns)
                for req_col in required_columns:
                    if req_col not in final_cols:
                        final_cols.append(req_col)
                return df.reindex(columns=final_cols)

        return df
    except pd.errors.EmptyDataError:
        # print(f"[{func_name}] Warning: File {file_path} is empty (EmptyDataError). Returning empty DataFrame.") # Reduce noise
        return (
            pd.DataFrame(columns=required_columns)
            if required_columns
            else pd.DataFrame()
        )
    except (
        FileNotFoundError
    ):  # Should be caught by exists() check, but belt-and-suspenders
        print(
            f"[{func_name}] Error: File not found at {file_path} (FileNotFoundError)."
        )
        return (
            pd.DataFrame(columns=required_columns)
            if required_columns
            else pd.DataFrame()
        )
    except Exception as e:
        print(f"[{func_name}] Error reading CSV file {file_path}: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging
        return (
            pd.DataFrame(columns=required_columns)
            if required_columns
            else pd.DataFrame()
        )


def load_transactions():
    """Loads and performs basic cleaning on transactions data."""
    func_name = "load_transactions"
    # --- CRITICAL ASSUMPTION ---
    # Assumes 'price' column in transactions.csv is the total transaction value.
    transaction_columns = [
        "trx_id",
        "user_id",
        "amount",
        "price",
        "type",
        "date",
        "ticker_id",
    ]
    # Load numeric columns as object initially to handle potential non-numeric values gracefully
    dtypes = {"trx_id": str, "user_id": str, "type": str, "date": str, "ticker_id": str}
    df = load_data(
        TRANSACTIONS_FILE, required_columns=transaction_columns, dtype=dtypes
    )

    if df.empty:
        # print(f"[{func_name}] Transaction data is empty or could not be loaded.") # Reduce noise
        return df  # Return the empty DataFrame (already has columns from load_data)

    # Coerce numeric types, invalid entries become NaN
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        # If 'price' column was missing despite being required (e.g., file manipulation)
        #  print(f"[{func_name}] CRITICAL WARNING: 'price' column missing from transactions. Cost basis will be incorrect.") # Reduce noise
        df["price"] = np.nan  # Add column as NaN

    # Convert date, invalid dates become NaT
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where essential data is missing AFTER coercion
    essential_cols = ["user_id", "ticker_id", "date", "amount", "price", "type"]
    original_rows = len(df)
    df.dropna(subset=essential_cols, inplace=True)
    # if len(df) < original_rows: # Reduce noise
    # print(f"[{func_name}] Dropped {original_rows - len(df)} rows due to missing essential data after coercion.")

    return df


def load_prices():
    """Loads and prepares historical price data."""
    func_name = "load_prices"
    price_columns = ["date", "ticker_id", "close_price"]
    prices_df = load_data(
        PRICES_FILE, required_columns=price_columns, dtype={"ticker_id": str}
    )

    if prices_df.empty:
        # print(f"[{func_name}] Historical price data is empty or could not be loaded.") # Reduce noise
        return pd.DataFrame(columns=["ticker_id", "market_price", "date"])

    # Rename early for clarity
    prices_df = prices_df.rename(
        columns={"close_price": "market_price", "date": "raw_date"}
    )

    # Convert date and price
    prices_df["date"] = pd.to_datetime(prices_df["raw_date"], errors="coerce")
    prices_df["market_price"] = pd.to_numeric(
        prices_df["market_price"], errors="coerce"
    )

    # Drop rows with invalid essential data
    original_rows = len(prices_df)
    prices_df.dropna(subset=["date", "ticker_id", "market_price"], inplace=True)
    # if len(prices_df) < original_rows: # Reduce noise
    #  print(f"[{func_name}] Dropped {original_rows - len(prices_df)} rows from prices due to missing essential data after coercion.")

    if prices_df.empty:
        # print(f"[{func_name}] Historical price data is empty after cleaning.") # Reduce noise
        return pd.DataFrame(columns=["ticker_id", "market_price", "date"])

    # Select final columns and remove duplicates for a ticker on the same date
    prices_final = prices_df[["ticker_id", "market_price", "date"]].drop_duplicates(
        subset=["ticker_id", "date"], keep="last"
    )  # Keep last in case of corrections

    # print(f"[{func_name}] Loaded {len(prices_final)} unique ticker-date price points.") # Reduce noise
    return prices_final


def calculate_portfolio_history(user_id: str):
    """
    Calculates detailed daily ticker history and aggregated portfolio summary.
    (Code from previous step - remains unchanged)
    """
    # print(f"\nCalculating portfolio history for user: {user_id}...") # Reduce noise
    # Define expected output structures for error cases first
    ticker_hist_cols = [
        "user_id",
        "ticker_id",
        "date",
        "net_quantity",
        "net_transaction_value",
        "cumulative_quantity",
        "cumulative_cost_basis",
        "market_price",
        "market_value",
    ]
    portfolio_summary_cols = [
        "user_id",
        "date",
        "daily_net_investment",
        "total_cost_basis",
        "total_market_value",
    ]
    empty_ticker_hist = pd.DataFrame(columns=ticker_hist_cols)
    empty_portfolio_summary = pd.DataFrame(columns=portfolio_summary_cols)

    # --- 1. Load Data ---
    transactions_df = load_transactions()
    prices_df = load_prices()  # Loads full historical prices

    if transactions_df.empty:
        # print("No transactions loaded. Cannot calculate history.") # Reduce noise
        return empty_ticker_hist, empty_portfolio_summary

    # --- 2. Preprocess User Transactions ---
    user_mask = transactions_df["user_id"] == str(user_id)
    user_trx = transactions_df[user_mask].copy()  # Use copy

    if user_trx.empty:
        # print(f"No transactions found for user {user_id}.") # Reduce noise
        return empty_ticker_hist, empty_portfolio_summary

    # Ensure data types and drop invalid rows
    user_trx["date"] = pd.to_datetime(user_trx["date"], errors="coerce")
    user_trx["amount"] = pd.to_numeric(user_trx["amount"], errors="coerce")
    user_trx["price"] = pd.to_numeric(user_trx["price"], errors="coerce")
    user_trx.dropna(
        subset=["date", "amount", "price", "ticker_id", "type"], inplace=True
    )

    if user_trx.empty:  # Check again after dropna
        # print(f"No valid transactions remaining for user {user_id} after cleaning.") # Reduce noise
        return empty_ticker_hist, empty_portfolio_summary

    # Calculate Net Quantity
    user_trx["net_quantity"] = user_trx["amount"]
    sell_mask = user_trx["type"].str.lower() == "sell"
    user_trx.loc[sell_mask, "net_quantity"] = -user_trx.loc[sell_mask, "net_quantity"]

    # Calculate Net Transaction Value
    user_trx["net_transaction_value"] = user_trx["price"]
    # UNCOMMENT IF price in CSV is always positive:
    # user_trx.loc[sell_mask, 'net_transaction_value'] = -user_trx.loc[sell_mask, 'net_transaction_value']
    user_trx.dropna(subset=["net_quantity", "net_transaction_value"], inplace=True)

    # Prepare for aggregation
    user_trx_processed = user_trx[
        ["ticker_id", "date", "net_quantity", "net_transaction_value"]
    ]
    user_trx_processed = user_trx_processed.sort_values(by=["ticker_id", "date"])

    # --- 3. Determine Date Range ---
    try:
        start_date = user_trx_processed["date"].min()
        end_date = (
            prices_df["date"].max()
            if (prices_df is not None and not prices_df.empty)
            else pd.Timestamp.now(tz="UTC")
        )

        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError("Invalid start/end date")
        start_date = start_date.tz_localize(None)
        end_date = end_date.tz_localize(None)
        if start_date > end_date:
            end_date = start_date

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        # print(f"Date range: {start_date.date()} to {end_date.date()}") # Reduce noise
        if date_range.empty:
            raise ValueError("Date range is empty")
    except Exception as e:
        print(f"Error setting date range: {e}. Cannot proceed.")
        return empty_ticker_hist, empty_portfolio_summary

    # --- 4. Calculate Daily Changes and Cumulative Holdings (Optimized) ---
    daily_changes = user_trx_processed.groupby(["ticker_id", "date"]).agg(
        net_quantity=("net_quantity", "sum"),
        net_transaction_value=("net_transaction_value", "sum"),
    )

    tickers = user_trx_processed["ticker_id"].unique()
    multi_index = pd.MultiIndex.from_product(
        [tickers, date_range], names=["ticker_id", "date"]
    )

    daily_history = daily_changes.reindex(multi_index, fill_value=0.0)

    daily_history["cumulative_quantity"] = daily_history.groupby(level="ticker_id")[
        "net_quantity"
    ].cumsum()
    daily_history["cumulative_cost_basis"] = daily_history.groupby(level="ticker_id")[
        "net_transaction_value"
    ].cumsum()

    ticker_daily_history = daily_history.reset_index()

    # --- 5. Merge Market Prices and Calculate Market Value ---
    if prices_df.empty:
        ticker_daily_history["market_price"] = np.nan
        ticker_daily_history["market_value"] = 0.0
    else:
        prices_to_merge = prices_df[["ticker_id", "date", "market_price"]]
        ticker_daily_history["date"] = pd.to_datetime(ticker_daily_history["date"])
        prices_to_merge["date"] = pd.to_datetime(prices_to_merge["date"])

        ticker_daily_history = pd.merge(
            ticker_daily_history, prices_to_merge, on=["ticker_id", "date"], how="left"
        )

        ticker_daily_history = ticker_daily_history.sort_values(
            by=["ticker_id", "date"]
        )
        ticker_daily_history["market_price"] = ticker_daily_history.groupby(
            "ticker_id"
        )["market_price"].ffill()
        ticker_daily_history["market_value"] = (
            ticker_daily_history["cumulative_quantity"]
            * ticker_daily_history["market_price"]
        )
        ticker_daily_history["market_value"] = ticker_daily_history[
            "market_value"
        ].fillna(0.0)

    # --- 6. Calculate Portfolio Summary ---
    portfolio_summary = (
        ticker_daily_history.groupby("date")
        .agg(
            daily_net_investment=("net_transaction_value", "sum"),
            total_cost_basis=("cumulative_cost_basis", "sum"),
            total_market_value=("market_value", "sum"),
        )
        .reset_index()
    )

    for col in ["daily_net_investment", "total_cost_basis", "total_market_value"]:
        portfolio_summary[col] = pd.to_numeric(
            portfolio_summary[col], errors="coerce"
        ).fillna(0.0)

    # --- 7. Finalize Output ---
    user_id_str = str(user_id)
    ticker_daily_history["user_id"] = user_id_str
    portfolio_summary["user_id"] = user_id_str

    final_ticker_hist = ticker_daily_history.reindex(columns=ticker_hist_cols)
    final_portfolio_summary = portfolio_summary.reindex(columns=portfolio_summary_cols)

    num_cols_ticker = [
        "net_quantity",
        "net_transaction_value",
        "cumulative_quantity",
        "cumulative_cost_basis",
        "market_value",
    ]
    num_cols_portfolio = [
        "daily_net_investment",
        "total_cost_basis",
        "total_market_value",
    ]

    for col in num_cols_ticker:
        if col in final_ticker_hist.columns:
            final_ticker_hist[col] = final_ticker_hist[col].fillna(0.0)
    for col in num_cols_portfolio:
        if col in final_portfolio_summary.columns:
            final_portfolio_summary[col] = final_portfolio_summary[col].fillna(0.0)

    # print(f"Calculation complete for user {user_id}.") # Reduce noise
    return final_ticker_hist, final_portfolio_summary


# ==============================================================================
# --- Dash Application Setup ---
# ==============================================================================

# --- Calculate Data ONCE for the specific user ---
TARGET_USER_ID = "1"
print(f"--- Pre-calculating data for User ID: {TARGET_USER_ID} ---")
try:
    _, portfolio_summary_data = calculate_portfolio_history(TARGET_USER_ID)
    if portfolio_summary_data.empty:
        print(
            f"WARNING: No data generated for user {TARGET_USER_ID}. Dashboard will be empty."
        )
    # Ensure date column is datetime if dataframe is not empty
    if not portfolio_summary_data.empty:
        portfolio_summary_data["date"] = pd.to_datetime(portfolio_summary_data["date"])

except Exception as e:
    print(f"ERROR: Failed to calculate initial data: {e}")
    portfolio_summary_data = pd.DataFrame(
        columns=[  # Create empty placeholder
            "user_id",
            "date",
            "daily_net_investment",
            "total_cost_basis",
            "total_market_value",
        ]
    )


# --- Initialize Dash App ---
app = dash.Dash(__name__)
app.title = f"Portfolio Dashboard (User: {TARGET_USER_ID})"  # Browser tab title

# --- App Layout ---
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.H1(
            f"Portfolio History Dashboard - User: {TARGET_USER_ID}"
        ),  # Main page title
        # --- Performance Display ---
        html.Div(
            [
                html.H3(
                    "Period Performance:",
                    style={"display": "inline-block", "margin-right": "10px"},
                ),
                html.Div(
                    id="profit-percentage-display",
                    children="Select a range or zoom...",
                    style={"display": "inline-block", "font-weight": "bold"},
                ),
            ]
        ),
        # --- Graph Section Title (Added) ---
        html.H3(
            "Cost Basis vs Market Value",
            style={"text-align": "center", "margin-top": "20px"},
        ),
        # --- Graph ---
        dcc.Graph(id="portfolio-graph", config={"displayModeBar": False}),
        # --- Data Store ---
        dcc.Store(
            id="full-data-range-store",
            data={
                "start": (
                    portfolio_summary_data["date"].min().isoformat()
                    if not portfolio_summary_data.empty
                    else None
                ),
                "end": (
                    portfolio_summary_data["date"].max().isoformat()
                    if not portfolio_summary_data.empty
                    else None
                ),
            },
        ),
    ]
)
# --- Callbacks ---


# Callback to generate the initial graph figure
@app.callback(
    Output("portfolio-graph", "figure"),
    Input("url", "pathname"),  # Use URL pathname as trigger for initial load
)
def update_graph(pathname):  # Input argument needed but not used here
    print("CALLBACK: update_graph triggered.")  # Add print statement
    if portfolio_summary_data.empty:
        print("   update_graph: No data, returning empty figure.")
        return go.Figure(layout=go.Layout(title="No data available"))

    print("   update_graph: Creating figure...")
    fig = go.Figure()

    # Add Cost Basis Line
    fig.add_trace(
        go.Scatter(
            x=portfolio_summary_data["date"],
            y=portfolio_summary_data["total_cost_basis"],
            mode="lines",
            name="Total Cost Basis",
            line=dict(color="royalblue", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Cost Basis: $%{y:,.2f}<extra></extra>",
        )
    )

    # Add Market Value Line
    fig.add_trace(
        go.Scatter(
            x=portfolio_summary_data["date"],
            y=portfolio_summary_data["total_market_value"],
            mode="lines",
            name="Total Market Value",
            line=dict(color="firebrick", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Market Value: $%{y:,.2f}<extra></extra>",
        )
    )

    # Update Layout WITHOUT internal title, keep Range Selector etc.
    fig.update_layout(
        # title=f'Cost Basis vs Market Value', # <<< REMOVE THIS LINE
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        legend_title_text="Metric",
        margin=dict(l=40, r=20, t=20, b=30),  # Adjust top margin if needed (t=20)
        xaxis=dict(
            rangeselector=dict(
                # ... (button configuration remains the same) ...
                buttons=list(
                    [
                        dict(count=5, label="5D", step="day", stepmode="backward"),
                        dict(count=1, label="MTD", step="month", stepmode="todate"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            ),
            rangeslider=dict(visible=False),  # Keep rangeslider visible
            type="date",
        ),
    )
    print("   update_graph: Figure created (without internal title).")
    return fig


# Callback to update the profit percentage based on visible range
@app.callback(
    Output("profit-percentage-display", "children"),
    Input("portfolio-graph", "relayoutData"),  # Listen to zoom/pan/button events
    State("full-data-range-store", "data"),  # Get the initial full range
)
def update_profit_percentage(relayout_data, full_range):
    print(
        f"CALLBACK: update_profit_percentage triggered. relayout_data: {relayout_data}"
    )  # DEBUG print

    # Default text if no range or data
    profit_text = "..."
    profit_color = "black"

    if portfolio_summary_data.empty or len(portfolio_summary_data) < 2:
        return html.Span("Not enough data", style={"color": profit_color})

    # Determine start and end dates from relayout_data or use full range
    start_date_str = None
    end_date_str = None
    update_type = "Initial Load/Default"  # Track how range was determined

    # Check if relayout_data exists and contains specific range keys
    if relayout_data:
        # Prioritize explicit range from zoom/pan
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            start_date_str = relayout_data["xaxis.range[0]"]
            end_date_str = relayout_data["xaxis.range[1]"]
            update_type = "Zoom/Pan"
        # Check for autorange (likely from 'All' button or reset)
        elif "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
            if full_range and full_range["start"] and full_range["end"]:
                start_date_str = full_range["start"]
                end_date_str = full_range["end"]
                update_type = "Autorange/Reset"
        # Fallback for other relayout events (could be button clicks not setting explicit range)
        # Or if specific keys like 'xaxis.range' are missing after button click
        elif full_range and full_range["start"] and full_range["end"]:
            start_date_str = full_range["start"]
            end_date_str = full_range["end"]
            update_type = (
                f"Fallback/Button? ({list(relayout_data.keys())})"  # Log keys found
            )

    else:
        # Initial load or no specific event data, use full range
        if full_range and full_range["start"] and full_range["end"]:
            start_date_str = full_range["start"]
            end_date_str = full_range["end"]
            update_type = "Initial Load/Default"

    # If we have valid date strings, calculate percentage
    if start_date_str and end_date_str:
        try:
            # Convert date strings (handle potential time/timezone info flexibly)
            # Using split might be fragile if format changes, pd.to_datetime is robust
            start_date = pd.to_datetime(start_date_str, errors="coerce").floor(
                "D"
            )  # Floor to start of day
            end_date = pd.to_datetime(end_date_str, errors="coerce").ceil(
                "D"
            )  # Ceil to end of day

            if pd.isna(start_date) or pd.isna(end_date):
                raise ValueError("Could not parse dates from relayoutData")

            # Filter the main dataframe based on the visible date range
            mask = (portfolio_summary_data["date"] >= start_date) & (
                portfolio_summary_data["date"] <= end_date
            )
            filtered_df = portfolio_summary_data.loc[mask]

            if (
                not filtered_df.empty and len(filtered_df) >= 1
            ):  # Need at least one point
                # Find the closest available data point to the theoretical start/end of the range
                # Sort by date to ensure iloc[0] and iloc[-1] are correct
                filtered_df = filtered_df.sort_values(by="date")
                start_value = filtered_df["total_market_value"].iloc[0]
                end_value = filtered_df["total_market_value"].iloc[-1]

                # Now calculate percentage based on these actual start/end values
                if (
                    pd.notna(start_value)
                    and pd.notna(end_value)
                    and abs(start_value) > 1e-9
                ):  # Avoid division by near-zero
                    profit_pct = ((end_value - start_value) / start_value) * 100
                    profit_text = f"{profit_pct:+.2f}%"
                    profit_color = (
                        "green" if profit_pct >= -0.005 else "red"
                    )  # Tolerate tiny negative as zero
                elif (
                    pd.notna(start_value)
                    and pd.notna(end_value)
                    and abs(start_value) <= 1e-9
                    and end_value > 1e-9
                ):
                    profit_text = "+Inf%"  # Started at zero, ended positive
                    profit_color = "green"
                elif (
                    pd.notna(start_value)
                    and pd.notna(end_value)
                    and abs(start_value) <= 1e-9
                    and abs(end_value) <= 1e-9
                ):
                    profit_text = "0.00%"  # Started and ended at zero
                    profit_color = "grey"
                elif (
                    pd.notna(start_value)
                    and pd.notna(end_value)
                    and abs(start_value) <= 1e-9
                    and end_value < -1e-9
                ):
                    profit_text = "-Inf%"  # Started at zero, ended negative
                    profit_color = "red"
                else:
                    profit_text = "N/A"  # Missing start or end value in filtered range
                    profit_color = "grey"
            else:  # No data in selected range
                profit_text = "N/A"
                profit_color = "grey"

        except Exception as e:
            print(f"Error processing relayout data or calculating percentage: {e}")
            profit_text = "Error"
            profit_color = "orange"
    else:
        # Case where start/end dates couldn't be determined
        profit_text = "..."
        profit_color = "black"

    print(
        f"   update_profit_percentage: Type='{update_type}', Range='{start_date_str}' to '{end_date_str}', Result='{profit_text}'"
    )
    # Return the result wrapped in a Span with dynamic color
    return html.Span(profit_text, style={"color": profit_color})


# --- Run the Dash App ---
if __name__ == "__main__":
    print("--- Starting Dash Server ---")
    # Set debug=False for production deployment
    app.run(debug=True)
