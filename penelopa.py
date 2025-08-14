import hashlib
import os
import sqlite3
import subprocess
import zlib
from datetime import UTC, datetime

DB_PATH = 'penelopa.db'
ORIGIN_TEXT = os.path.join('origin', 'molly.md')
THRESHOLD_BYTES = 100 * 1024  # 100 kilobytes


def get_current_commit() -> str:
    """Return the current git commit hash."""
    return (
        subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        .decode('utf-8')
        .strip()
    )


def repo_sha256(commit_hash: str) -> str:
    """Return a SHA-256 digest for the given commit hash."""
    return hashlib.sha256(commit_hash.encode('utf-8')).hexdigest()


def get_diff(prev_commit: str, current_commit: str) -> str:
    """Return git diff between two commits."""
    if prev_commit:
        diff_cmd = ['git', 'diff', prev_commit, current_commit]
        return subprocess.check_output(diff_cmd).decode('utf-8')
    # No previous commit tracked: show commit itself
    show_cmd = ['git', 'show', current_commit]
    return subprocess.check_output(show_cmd).decode('utf-8')


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_hash TEXT,
            repo_hash TEXT,
            diff BLOB,
            size INTEGER,
            created_at TEXT
        )
        """
    )
    conn.commit()


def get_last_commit(conn: sqlite3.Connection) -> str | None:
    cur = conn.cursor()
    cur.execute('SELECT commit_hash FROM changes ORDER BY id DESC LIMIT 1')
    row = cur.fetchone()
    return row[0] if row else None


def log_change(
    conn: sqlite3.Connection,
    commit_hash: str,
    repo_hash: str,
    diff: str,
) -> None:
    compressed_diff = zlib.compress(diff.encode('utf-8'))
    size = len(compressed_diff)
    conn.execute(
        '''
        INSERT INTO changes (commit_hash, repo_hash, diff, size, created_at)
        VALUES (?, ?, ?, ?, ?)
        ''',
        (
            commit_hash,
            repo_hash,
            compressed_diff,
            size,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()


def fetch_diff(conn: sqlite3.Connection, commit_hash: str) -> str | None:
    """Return the decompressed diff for a commit hash if present."""
    cur = conn.cursor()
    cur.execute(
        'SELECT diff FROM changes WHERE commit_hash = ?',
        (commit_hash,),
    )
    row = cur.fetchone()
    if row:
        return zlib.decompress(row[0]).decode('utf-8')
    return None


def total_logged_size(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute('SELECT COALESCE(SUM(size), 0) FROM changes')
    return cur.fetchone()[0]


def cleanup_db(conn: sqlite3.Connection) -> None:
    """Delete oldest rows until total size is under the threshold."""
    while total_logged_size(conn) > THRESHOLD_BYTES:
        conn.execute(
            """
            DELETE FROM changes WHERE id = (
                SELECT id FROM changes ORDER BY id ASC LIMIT 1
            )
            """
        )
    conn.commit()


def fine_tune() -> None:
    """Trigger fine-tuning on Molly's monologue."""
    print('Fine-tuning triggered on Molly\'s monologue...')
    # Placeholder: integrate with nanoGPT training as needed
    # subprocess.run(
    #     ['python', 'train.py', '--dataset', ORIGIN_TEXT],
    #     check=True,
    # )


def main() -> None:
    commit = get_current_commit()
    sha = repo_sha256(commit)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    last_commit = get_last_commit(conn)
    if commit != last_commit:
        diff = get_diff(last_commit, commit)
        log_change(conn, commit, sha, diff)
        print('Repository change detected and logged.')
    else:
        print('No repository changes detected.')

    if total_logged_size(conn) > THRESHOLD_BYTES:
        fine_tune()
        cleanup_db(conn)

    conn.close()


if __name__ == '__main__':
    main()
