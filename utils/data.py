from io import BytesIO
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Datasets shipped with the app (shown under Data source -> Available data).
# to add another one, copy the file into the data or folder and add a line here.
AVAILABLE_DATA = {
    "Diabetes data": DATA_DIR / "diabetes.csv",
}

SUPPORTED = ["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"]
EXCEL = (".xlsx", ".xls")


def is_excel(name: str) -> bool:
    return name.lower().endswith(EXCEL)


def excel_sheets(source) -> list:
    """Sheet names of an Excel file (path or file-like)."""
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.ExcelFile(source).sheet_names


def read_any(source, name: str, sheet=None) -> pd.DataFrame:
    """Read a CSV / Excel / JSON / TXT / Parquet file from a path or file-like object."""
    if hasattr(source, "seek"):
        source.seek(0)
    n = name.lower()
    if n.endswith(".csv"):
        return pd.read_csv(source)
    if n.endswith(".tsv"):
        return pd.read_csv(source, sep="\t")
    if n.endswith(".txt"):
        return pd.read_csv(source, sep=None, engine="python")
    if n.endswith(EXCEL):
        return pd.read_excel(source, sheet_name=sheet or 0)
    if n.endswith(".json"):
        return pd.read_json(source)
    if n.endswith(".parquet"):
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported file type: {name}")


def load_available(label: str) -> pd.DataFrame:
    """Load one of the built-in datasets listed in AVAILABLE_DATA."""
    path = AVAILABLE_DATA[label]
    return read_any(path, path.name)



def load_url(url: str) -> pd.DataFrame:
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("The URL must start with http:// or https://")
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    clean = url.split("?")[0].split("#")[0]
    return read_any(BytesIO(r.content), clean if "." in Path(clean).name else "data.csv")


def combine(frames: dict) -> pd.DataFrame:
    """Stack several files; adds a 'source_file' column."""
    parts = []
    for name, df in frames.items():
        df = df.copy()
        df["source_file"] = name
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ---- Databases ----
def database_engine(kind: str, v: dict):
    """Create a database connection. A missing database connects to the server catalog."""
    if kind == "SQLite":
        path = str(Path(v["path"]).expanduser())
        return create_engine(f"sqlite:///{path}")
    port = int(v["port"]) if str(v.get("port", "")).strip() else None
    host = (v.get("host") or "localhost").strip()
    database = v.get("database")
    if kind == "MySQL":
        return create_engine(URL.create(
            drivername="mysql+pymysql", username=v.get("user"), password=v.get("password"),
            host=host, port=port or 3306, database=database
        ), pool_pre_ping=True)
    if kind == "PostgreSQL":
        # PostgreSQL needs a database even to connect to its server catalogue.
        return create_engine(URL.create(
            drivername="postgresql+psycopg2", username=v.get("user"), password=v.get("password"),
            host=host, port=port or 5432, database=database or "postgres"
        ), pool_pre_ping=True)
    if kind == "SQL Server":
        driver = (v.get("driver") or "ODBC Driver 17 for SQL Server").strip()
        target_db = database or "master"
        conn = (f"DRIVER={{{driver}}};SERVER={host},{port or 1433};DATABASE={target_db};"
                f"UID={v.get('user', '')};PWD={v.get('password', '')};TrustServerCertificate=yes")
        return create_engine("mssql+pyodbc:///?odbc_connect=" + quote_plus(conn), pool_pre_ping=True)
    raise ValueError("Unsupported database.")


def get_databases(engine) -> list:
    """List databases visible to the connected account."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        if dialect == "mysql":
            rows = conn.exec_driver_sql("SHOW DATABASES").fetchall()
            names = [str(row[0]) for row in rows]
            return [n for n in names if n not in {"information_schema", "performance_schema", "mysql", "sys"}]
        if dialect == "postgresql":
            rows = conn.exec_driver_sql("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname").fetchall()
            return [str(row[0]) for row in rows]
        if dialect == "mssql":
            rows = conn.exec_driver_sql("SELECT name FROM sys.databases WHERE state_desc = 'ONLINE' ORDER BY name").fetchall()
            return [str(row[0]) for row in rows if str(row[0]) not in {"master", "model", "msdb", "tempdb"}]
        return []

def get_tables(engine) -> list:
    insp = inspect(engine)
    return insp.get_table_names() + insp.get_view_names()


def load_table(engine, table: str, limit: int = 200_000) -> pd.DataFrame:
    """Load a safely quoted table or view, with a row limit."""
    preparer = engine.dialect.identifier_preparer
    quoted = preparer.quote(table)
    q = f"SELECT * FROM {quoted}"
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn).head(limit)

def run_query(engine, sql: str) -> pd.DataFrame:
    if not sql.strip().lower().startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed.")
    with engine.connect() as c:
        return pd.read_sql(text(sql), c)
