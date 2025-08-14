# Logging and Backup

This project stores runtime logs in the `logs/` directory, which is created
with permissions `700` so only the owner can access it.

To back up the SQLite database (`lines.db`) and the text log (`lines.txt`), use
`backup_logs.sh`:

```bash
./backup_logs.sh /path/to/secure/backup
```

Schedule this script with `cron` or another task scheduler to run periodically
and safeguard your data.
