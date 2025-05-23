# src/your_package_name/sql_models.py

import enum
from typing import List, Optional
from datetime import datetime, timezone

from sqlmodel import Field, Relationship, SQLModel, create_engine, Session
from sqlalchemy.sql import func  # For server-side default for updated_at
from sqlalchemy import Column, DateTime as SADateTime  # For onupdate


# --- Enums ---
# This enum can be used directly by SQLModel
class TransactionCategory(str, enum.Enum):
    """
    Enum for the type of transaction (buy or sell).
    """

    BUY = "buy"
    SELL = "sell"


# --- Database Setup (Example - typically in a database.py or main.py) ---
# DATABASE_URL = "postgresql://user:password@host:port/yourdbname" # Replace with your actual URL
# engine = create_engine(DATABASE_URL, echo=True) # echo=True for debugging SQL

# def create_db_and_tables():
#     SQLModel.metadata.create_all(engine)

# def get_session():
#     with Session(engine) as session:
#         yield session

# --- Forward References for Relationships ---
# SQLModel handles forward references for type hints automatically if models are in the same file
# or if you use string literals and update_forward_refs() later.


# --- Ticker Model ---
class TickerBase(SQLModel):
    """
    Base for Ticker, used for creation and common fields.
    Not a table model itself.
    """

    symbol: str = Field(
        ...,
        max_length=10,
        unique=True,
        index=True,
        description="Ticker symbol (e.g., AAPL)",
    )
    name: str = Field(..., max_length=100, description="Full name of the company/asset")


class Ticker(TickerBase, table=True):
    """
    Ticker table model. Inherits from TickerBase.
    """

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    # For onupdate, we need to use sa_column with SQLAlchemy's Column capabilities
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SADateTime(timezone=True), onupdate=func.now(), default=func.now()
        ),
        nullable=False,
    )

    # Relationship: A ticker can have many transactions
    transactions: List["Transaction"] = Relationship(back_populates="ticker")


class TickerCreate(TickerBase):
    """
    Schema for creating a Ticker.
    """

    pass


class TickerRead(TickerBase):
    """
    Schema for reading a Ticker (excluding relationships by default).
    """

    id: int
    created_at: datetime
    updated_at: datetime


class TickerReadWithTransactions(TickerRead):
    """
    Schema for reading a Ticker along with its transactions.
    """

    transactions: List["TransactionRead"] = []


# --- User Model ---
class UserBase(SQLModel):
    """
    Base for User, used for creation and common fields.
    """

    email: str = Field(..., unique=True, index=True, max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=100)


class User(UserBase, table=True):
    """
    User table model.
    """

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    hashed_password: str = Field(nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SADateTime(timezone=True), onupdate=func.now(), default=func.now()
        ),
        nullable=False,
    )

    # Relationship: A user can have many transactions
    transactions: List["Transaction"] = Relationship(back_populates="owner")


class UserCreate(UserBase):
    """
    Schema for creating a User. Includes password.
    """

    password: str = Field(
        ..., min_length=8, description="User password, must be at least 8 characters"
    )


class UserRead(UserBase):
    """
    Schema for reading a User (sensitive info like password hash is excluded as it's not in UserBase).
    """

    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserReadWithTransactions(UserRead):
    """
    Schema for reading a User along with their transactions.
    """

    transactions: List["TransactionRead"] = []


# --- Transaction Model ---
class TransactionBase(SQLModel):
    """
    Base for Transaction, used for creation and common fields.
    """

    qty: int = Field(..., gt=0, description="Quantity of the asset transacted")
    price: int = Field(
        ..., gt=0, description="Price per unit at the time of transaction"
    )
    category: TransactionCategory  # Uses the TransactionCategory enum
    transaction_date: datetime = Field(
        ..., description="Date and time of the transaction"
    )
    # Foreign keys will be on the table model


class Transaction(TransactionBase, table=True):
    """
    Transaction table model.
    """

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SADateTime(timezone=True), onupdate=func.now(), default=func.now()
        ),
        nullable=False,
    )

    # Foreign Keys
    ticker_id: int = Field(..., foreign_key="ticker.id", index=True)
    user_id: int = Field(..., foreign_key="user.id", index=True)

    # Relationships
    owner: Optional[User] = Relationship(back_populates="transactions")
    ticker: Optional[Ticker] = Relationship(back_populates="transactions")


class TransactionCreate(TransactionBase):
    """
    Schema for creating a Transaction.
    user_id will typically come from the authenticated user context in the API endpoint.
    ticker_id is required to associate the transaction with a ticker.
    """

    user_id: int  # Explicitly required for creation if not derived from auth
    ticker_id: int


class TransactionRead(TransactionBase):
    """
    Schema for reading a Transaction.
    """

    id: int
    user_id: int
    ticker_id: int  # Already in TransactionBase, but good to be explicit for read model
    created_at: datetime
    updated_at: datetime


class TransactionReadWithDetails(TransactionRead):
    """
    Schema for reading a Transaction with its related owner (User) and ticker (Ticker).
    """

    owner: Optional[UserRead] = None  # Use UserRead to avoid sending password hash
    ticker: Optional[TickerRead] = None


# If you had models in different files or complex circular dependencies,
# you might need to call update_forward_refs() for each model at the end of the file
# where all relevant models are defined.
# However, SQLModel often handles this well if types are string literals or all in one scope.
# Example:
# Ticker.model_rebuild()
# User.model_rebuild()
# Transaction.model_rebuild()
# (SQLModel uses model_rebuild internally, similar to Pydantic's update_forward_refs)
# For SQLModel, direct calls to model_rebuild or update_forward_refs are less often needed
# if using string literals for forward references like List["Transaction"].
