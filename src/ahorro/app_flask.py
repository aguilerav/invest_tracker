import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash, generate_password_hash
import pandas as pd
from ahorro.utils.paths import DATA_DIR, TEMPLATE_DIR
from pathlib import Path

# --- Configuration ---
# Define the project root directory relative to this file (app.py)
# Assumes app.py is in the project root. Adjust if needed.
USERS_FILE = DATA_DIR / "users.csv"
CLIENTS_FILE = DATA_DIR / "clients.csv"

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
# Secret key is needed for session management and flash messages
# Change this to a random, secret value in a real application!
app.secret_key = "pass123"  # NEVER commit this directly in real projects


# --- Helper Functions ---
def load_users():
    """Loads user data from the CSV file."""
    try:
        return pd.read_csv(USERS_FILE)
    except FileNotFoundError:
        print(f"Error: Users file not found at {USERS_FILE}")
        return None
    except Exception as e:
        print(f"Error reading users CSV: {e}")
        return None


def load_clients():
    """Loads client data from the CSV file."""
    try:
        return pd.read_csv(CLIENTS_FILE)
    except FileNotFoundError:
        print(f"Error: Clients file not found at {CLIENTS_FILE}")
        return None
    except Exception as e:
        print(f"Error reading clients CSV: {e}")
        return None


def verify_user(username, password):
    """
    Verifies username and password hash against the users CSV.
    Returns the user_id (as standard Python int) if valid, otherwise None.
    Uses secure password hashing (Werkzeug).
    """
    users_df = load_users()
    if users_df is None:
        print("Error: Could not load user data.")
        return None

    # Find the user by username (comparing as strings)
    user_record = users_df[users_df["username"].astype(str) == str(username)]

    if not user_record.empty:
        # Retrieve the stored password HASH
        stored_hash = user_record.iloc[0]["pwd"]

        # Check if the provided password matches the stored hash
        # check_password_hash handles the salt automatically if stored with the hash (Werkzeug default)
        if stored_hash and check_password_hash(stored_hash, password):
            # Password hash matches! Retrieve user_id.
            user_id_from_df = user_record.iloc[0]["user_id"]
            # Convert numpy.int64 (or other types) to standard Python int for JSON serialization
            try:
                return int(user_id_from_df)
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Could not convert user_id '{user_id_from_df}' to int: {e}"
                )
                return None  # Or handle error appropriately
        else:
            # Password hash does not match or hash is missing
            print(f"Password check failed for user: {username}")
            return None
    else:
        # User not found
        print(f"User not found: {username}")
        return None


def get_client_name(user_id):
    """Gets the client name associated with a user_id."""
    clients_df = load_clients()
    if clients_df is None:
        return "Unknown Client"

    client_record = clients_df[clients_df["user_id"] == user_id]

    if not client_record.empty:
        return client_record.iloc[0]["client_name"]
    return "Client Not Found"  # Or handle cases where user might not have a client


# --- Flask Routes ---
@app.route("/")
def index():
    """Redirects to login page if not logged in, else to welcome page."""
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

        if user_id:
            # Login successful
            client_name = get_client_name(user_id)
            session["user_id"] = user_id  # Store user_id in session
            session["username"] = username
            session["client_name"] = client_name
            flash(f"Login successful! Welcome {client_name}.", "success")
            return redirect(url_for("welcome"))
        else:
            # Login failed
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

    # For GET request, just show the login page
    # If already logged in, redirect to welcome
    if "user_id" in session:
        return redirect(url_for("welcome"))
    return render_template("login.html")


@app.route("/welcome")
def welcome():
    """Displays the welcome page if logged in."""
    if "client_name" not in session:
        # Not logged in, redirect to login
        flash("You need to login first.", "warning")
        return redirect(url_for("login"))

    # Retrieve client name from session
    client_name = session.get(
        "client_name", "Guest"
    )  # Default to 'Guest' if somehow not set
    return render_template("welcome.html", client_name=client_name)


@app.route("/logout")
def logout():
    """Logs the user out by clearing the session."""
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("client_name", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# --- Run the App ---
if __name__ == "__main__":
    # Use host='0.0.0.0' to make it accessible on your local network
    # debug=True is helpful for development but should be False in production
    app.run(debug=True, host="0.0.0.0", port=5000)
