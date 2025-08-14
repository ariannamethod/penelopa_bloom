Molly Bloom is a Telegram bot that speaks in a looping monologue. The bot uses a small GPT model defined in `penelopa.py` and can be trained with the tools in `bloom.py`.

To deploy on Railway you only need a Telegram bot token. Copy `.env.example` to `.env` and fill in `TELEGRAM_TOKEN`. Install the packages from `requirements.txt` if you run the bot locally.

On Railway create a new project and link this repository. In the project settings add the `TELEGRAM_TOKEN` variable and deploy. Railway reads `Procfile` and runs `python molly.py` automatically.

After deployment the bot will start polling Telegram. You can also fine‑tune the model by running `python bloom.py` in the Railway shell or on your own machine.
