import sqlite3
from collections import defaultdict, namedtuple

Message = namedtuple("Message", [
    "chat_id",
    "rowid",
    "guid",
    "text",
    "is_from_me",
    "date",
])

def get_one_to_one_chat_ids(cursor):
    cursor.execute("""
        SELECT c.ROWID AS chat_id
        FROM chat c
        JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
        GROUP BY c.ROWID
        HAVING COUNT(DISTINCT chj.handle_id) = 1
    """)
    return {row[0] for row in cursor.fetchall()}


def get_clean_messages(cursor, chat_ids):
    placeholder = ",".join("?" for _ in chat_ids)
    sql = f"""
        SELECT cmj.chat_id, m.ROWID, m.guid, m.text, m.is_from_me, m.date,
               m.associated_message_guid, m.message_action_type, m.item_type,
               m.is_system_message, m.is_service_message, m.is_corrupt,
               m.thread_originator_guid, m.thread_originator_part
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE cmj.chat_id IN ({placeholder})
        ORDER BY cmj.chat_id, m.date, m.ROWID
    """
    cursor.execute(sql, tuple(chat_ids))
    messages = []
    for row in cursor.fetchall():
        (chat_id, rowid, guid, text, is_from_me, date,
         associated_message_guid, message_action_type, item_type,
         is_system_message, is_service_message, is_corrupt,
         thread_originator_guid, thread_originator_part) = row

        # filter for clean text messages only
        if not text or not str(text).strip():
            continue
        if is_system_message or is_service_message or is_corrupt:
            continue
        if associated_message_guid or message_action_type not in (None, 0) or item_type not in (None, 0):
            continue
        if thread_originator_guid or thread_originator_part:
            continue

        messages.append(Message(chat_id, rowid, guid, text.strip(), is_from_me, date))
    return messages


def pair_messages(messages):
    pairs = []
    by_chat = defaultdict(list)
    for msg in messages:
        by_chat[msg.chat_id].append(msg)

    for chat_id, msgs in by_chat.items():
        for i in range(len(msgs) - 1):
            curr, nxt = msgs[i], msgs[i + 1]
            if curr.is_from_me == 0 and nxt.is_from_me == 1:
                pairs.append((chat_id, curr.text, nxt.text, nxt.date))
    return pairs


def save_pairs_to_new_db(pairs, output_path="new_chat.db"):
    conn = sqlite3.connect(output_path)
    c = conn.cursor()

    # Create table for pairs
    c.execute("""
        CREATE TABLE IF NOT EXISTS paired_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER,
            insert_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            date INTEGER
        )
    """)

    c.executemany("""
        INSERT INTO paired_texts (chat_id, insert_text, response_text, date)
        VALUES (?, ?, ?, ?)
    """, pairs)

    conn.commit()
    conn.close()
    print(f"✅ Stored {len(pairs)} paired messages in {output_path}")


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python export_to_new_chatdb.py /path/to/chat.db")
        sys.exit(1)

    db_path = sys.argv[1]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    one_to_one = get_one_to_one_chat_ids(cursor)
    messages = get_clean_messages(cursor, one_to_one)
    pairs = pair_messages(messages)

    save_pairs_to_new_db(pairs)
    conn.close()


if __name__ == "__main__":
    main()
