Molly is a simple Telegram bot powered by the Penelopa language model. The code was cleaned to three Python files: `molly.py` for the bot, `penelopa.py` for the model and `bloom.py` for training.

To run locally, copy `.env.example` to `.env` and put your `TELEGRAM_TOKEN`. Install the packages with `pip install -r requirements.txt` and start the bot using `python molly.py`.

Training or finetuning happens with `bloom.py`. Place your tokenized dataset in `data/<name>/train.bin` and `val.bin`, then run `python bloom.py --dataset <name>`.

For Railway deployment, push this repo to your account and create a new project from it. Add the `TELEGRAM_TOKEN` variable in the dashboard, and Railway will run `python molly.py` defined in the `Procfile`.
