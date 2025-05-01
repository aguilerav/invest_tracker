import pandas as pd
from werkzeug.security import generate_password_hash
from ahorro.utils.paths import DATA_DIR
import sys

# --- Configuration ---
# Input file with plain text passwords
PLAIN_USERS_FILE = DATA_DIR / "users_plain.csv"
# Output file with hashed passwords
HASHED_USERS_FILE = DATA_DIR / "users.csv"


# --- Main Script Logic ---
def hash_passwords_in_csv(input_file, output_file):
    """
    Reads a CSV file with user data, hashes the passwords in the 'pwd' column,
    and saves the result to a new CSV file.
    """
    print(f"Attempting to read plain text user data from: {input_file}")
    try:
        # Read the CSV with plain text passwords
        # Ensure columns are read as strings initially to avoid type issues
        users_df = pd.read_csv(input_file, dtype=str)
        print(f"Successfully read {len(users_df)} records from {input_file}")

    except FileNotFoundError:
        print(
            f"Error: Input file not found at '{input_file}'. Please ensure the file exists."
        )
        sys.exit(1)  # Exit the script with an error code
    except Exception as e:
        print(f"Error reading input CSV file '{input_file}': {e}")
        sys.exit(1)

    # Check if 'pwd' column exists
    if "pwd" not in users_df.columns:
        print(f"Error: Required column 'pwd' not found in '{input_file}'.")
        sys.exit(1)

    print("Hashing passwords...")
    # Apply the hashing function to the 'pwd' column
    # generate_password_hash creates a hash including the method and salt
    try:
        # Use .astype(str) just in case some passwords were read as numbers/other types
        users_df["pwd"] = users_df["pwd"].astype(str).apply(generate_password_hash)
        print("Password hashing complete.")
    except Exception as e:
        print(f"Error during password hashing: {e}")
        sys.exit(1)

    print(f"Attempting to save hashed user data to: {output_file}")
    try:
        # Save the DataFrame with hashed passwords to the new CSV file
        # index=False prevents pandas from writing the DataFrame index as a column
        users_df.to_csv(output_file, index=False)
        print(f"Successfully saved hashed user data to '{output_file}'.")

    except Exception as e:
        print(f"Error writing output CSV file '{output_file}': {e}")
        sys.exit(1)


# --- Run the Script ---
if __name__ == "__main__":
    # Ensure the data directory exists (optional, but good practice)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Call the main function
    hash_passwords_in_csv(PLAIN_USERS_FILE, HASHED_USERS_FILE)
    print("\nScript finished.")
