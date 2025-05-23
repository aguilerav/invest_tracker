import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import (
    Enum as SQLAlchemyEnum,
)
from .database import Base


# Enums
class TransactionCategory(enum.Enum):
    """
    Enum for the type of transaction (buy or sell).
    """

    BUY = "buy"
    SELL = "sell"


# Models
class User(Base):
    __tablename__ = "users"

    # Columns for the 'users' table
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # One-to-many relationship: A user can have many transactions.
    transactions = relationship(
        "Transaction", back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self):
        # A string representation of the User object, useful for debugging.
        return (
            f"<User(id={self.id}, email='{self.email}', full_name='{self.full_name}')>"
        )


# Transactions
class Transactions(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    qty = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    category = Column(
        SQLAlchemyEnum(TransactionCategory, name="transaction_category_enum"),
        nullable=False,
    )
    transaction_date = Column(DateTime(timezone=True), server_default=func.now())
    ticker_id = Column(Integer, ForeignKey("ticker.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner = relationship("User", back_populates="transactions")
    ticker = relationship("Ticker", back_populates="transactions")

    def __repr__(self):
        # A string representation of the Transaction object, useful for debugging.
        return (
            f"<Transaction(id={self.id}, user_id={self.user_id}, ticker_id={self.ticker_id}, "
            f"type='{self.type.value if self.type else None}', qty={self.qty}, price={self.price}, "
            f"date='{self.transaction_date}')>"
        )


# Ticker
class Ticker(Base):
    __tablename__ = "ticker"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    symbol = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    transactions = relationship("Transactions", back_populates="ticker")

    def __repr__(self):
        return f"<Ticker(id={self.id}, symbol='{self.symbol}', name='{self.name}')>"
