from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from agent_platform.core.config import get_settings


def create_database_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )


def check_database(engine: Engine) -> bool:
    with engine.connect() as connection:
        return connection.execute(text("SELECT 1")).scalar_one() == 1


def check_vector_extension(engine: Engine) -> bool:
    statement = text(
        "SELECT EXISTS("
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        ")"
    )
    with engine.connect() as connection:
        return bool(connection.execute(statement).scalar_one())