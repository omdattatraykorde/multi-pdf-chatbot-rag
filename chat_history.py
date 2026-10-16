import sqlite3
from datetime import datetime


DB_NAME = "chat_history.db"


def initialize_database():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            model TEXT,

            question TEXT,

            answer TEXT
        )
    """)

    conn.commit()

    conn.close()


def save_chat(model, question, answer):

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""

        INSERT INTO chat_history

        (timestamp, model, question, answer)

        VALUES (?, ?, ?, ?)

    """, (

        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        model,

        question,

        answer

    ))

    conn.commit()

    conn.close()


def load_chat_history():

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""

        SELECT
            id,
            timestamp,
            model,
            question,
            answer

        FROM chat_history

        ORDER BY id DESC

    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


def delete_chat(chat_id):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("DELETE FROM chat_history WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()