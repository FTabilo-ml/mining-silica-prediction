import os
from typing import Optional
from .settings import settings

_conn = None

def get_conn() -> Optional[object]:
    global _conn
    if not settings.DB_CONN:
        return None
    if _conn is None:
        # Ejemplos:
        # import psycopg2; _conn = psycopg2.connect(settings.DB_CONN)
        # import pyodbc; _conn = pyodbc.connect(settings.DB_CONN)
        pass
    return _conn
