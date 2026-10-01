import sqlite3
from typing import Dict, Any
from fastmcp import FastMCP

# 1. Initialize the FastMCP server
mcp = FastMCP("Enterprise CRM")

# 2. Setup a mock SQLite database in memory
def setup_mock_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            name TEXT,
            account_status TEXT,
            mrr REAL
        )
    ''')
    # Insert mock enterprise data
    users = [
        ("USR-001", "Acme Corp", "Active", 15000.00),
        ("USR-002", "Globex", "Churned", 0.00),
        ("USR-003", "Stark Industries", "Active", 45000.00)
    ]
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users)
    conn.commit()
    return conn

# Initialize DB globally for the tools to access
db_conn = setup_mock_db()

# 3. Expose functions as MCP Tools
@mcp.tool
def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Fetch the full enterprise profile for a given user ID."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    
    if row:
        return {"id": row[0], "name": row[1], "status": row[2], "mrr": row[3]}
    return {"error": f"User {user_id} not found."}

@mcp.tool
def check_active_accounts() -> list[str]:
    """Returns a list of names for all currently active enterprise accounts."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM users WHERE account_status = 'Active'")
    return [row[0] for row in cursor.fetchall()]

if __name__ == "__main__":
    # 4. Run the server using the standard STDIO transport for local agents
    print("Starting CRM MCP Server on STDIO...")
    mcp.run()