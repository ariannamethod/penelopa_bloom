import sqlite3

import penelopa


def test_log_change_inserts_row():
    conn = sqlite3.connect(":memory:")
    penelopa.init_db(conn)
    penelopa.log_change(conn, "abc", "sha", "diff")
    cur = conn.cursor()
    cur.execute(
        "SELECT commit_hash, repo_hash, diff, size FROM changes"
    )
    row = cur.fetchone()
    assert row == ("abc", "sha", "diff", len("diff".encode("utf-8")))
    assert cur.fetchone() is None
    conn.close()
