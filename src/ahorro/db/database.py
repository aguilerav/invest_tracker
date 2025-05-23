import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from ahorro.core.config import settings

DATABASE_URL = settings.DATABASE_URL
# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # Number of connections to keep in the pool
    max_overflow=20,  # Maximum number of connections to allow beyond pool_size
    poolclass=QueuePool,  # Use QueuePool for more robust connection handling
)

# Create a SessionLocal class
# autocommit=False: Trxs are not automatically committed. You need to call `db.commit()`.
# autoflush=False: Changes are not automatically flushed to the database. You control when to flush.
# bind=engine: Associates this session factory with our database engine.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Create a Base class for your declarative models
# This will be the base class for all your SQLAlchemy models
# Base = declarative_base()


# Helper function to get a database session
# This is useful for dependency injection in web frameworks like FastAPI
# def get_db():
#    db = SessionLocal()
#    try:
#        yield db
#    finally:
#        db.close()


# Optional: Function to create all tables defined in models.py
# You might use this for initial setup or testing, but for production,
# using a migration tool like Alembic is recommended.
# def create_all_tables():
#    Base.metadata.create_all(bind=engine)
