# Restore PostgreSQL databases and run ForecastBot

The demo and new experiments require all three databases below. Restore them before starting the application.

| Backup file | Database to create |
|---|---|
| `forecastbot.dump` | `forecastbot` |
| `m5_db.dump` | `m5_db` |
| `hcmcbus_db.dump` | `hcmcbus_db` |

The files are provided in this directory in the complete local distribution. If you downloaded the code from GitHub, download the three dump assets from the repository's Release and place them in `database/`. Dumps are excluded from ordinary Git commits. `manifest.json` records the file sizes and SHA-256 checksums.

## 1. Prepare PostgreSQL

Install and start PostgreSQL, and make sure `psql`, `createuser`, `createdb`, and `pg_restore` are available on your PATH. These backups were created using PostgreSQL client tools 14.17; PostgreSQL 14 with matching client tools reproduces that environment.

```bash
pg_restore --version
pg_isready -h localhost -p 5432
```

Run the remaining commands from the `ForecastBot` project root. The examples assume the administrator role is `postgres`; replace it with your actual administrator role if necessary. Homebrew installations may use your macOS account name.

## 2. Create a login role and three empty databases

If the role `forecastbot_user` does not already exist, create it:

```bash
createuser -h localhost -p 5432 -U postgres -W --pwprompt forecastbot_user
```

Set a password for the new role when prompted. The administrator password may also be requested. Use this new role and its password in `.env` later.

Create the databases:

```bash
createdb -h localhost -p 5432 -U postgres -W -O forecastbot_user -T template0 forecastbot
createdb -h localhost -p 5432 -U postgres -W -O forecastbot_user -T template0 m5_db
createdb -h localhost -p 5432 -U postgres -W -O forecastbot_user -T template0 hcmcbus_db
```

Restore into empty databases. If those names already contain data, use different database names and update both the restore commands and `.env`. Do not restore over an existing working database.

## 3. Restore the three dumps

Enter the password for `forecastbot_user` when prompted:

```bash
pg_restore -h localhost -p 5432 -U forecastbot_user -W --no-owner --no-privileges --single-transaction --exit-on-error -d forecastbot database/forecastbot.dump
pg_restore -h localhost -p 5432 -U forecastbot_user -W --no-owner --no-privileges --single-transaction --exit-on-error -d m5_db database/m5_db.dump
pg_restore -h localhost -p 5432 -U forecastbot_user -W --no-owner --no-privileges --single-transaction --exit-on-error -d hcmcbus_db database/hcmcbus_db.dump
```

The commands skip source ownership and access grants, create objects under `forecastbot_user`, and roll back the affected database's restore if an error occurs. They do not create the databases themselves. Allow enough disk space for uncompressed data and PostgreSQL's write-ahead log.

With pgAdmin, create the same empty databases, then right-click each database → **Restore** → select its `.dump` file using **Custom or tar** format. Disable restoration of ownership and privileges, and restore using the target database owner.

## 4. Check the restored data

```bash
psql -h localhost -p 5432 -U forecastbot_user -W -d forecastbot -c "SELECT dataset_code, count(*) FROM schema_knowledge WHERE is_active = TRUE GROUP BY dataset_code;"
psql -h localhost -p 5432 -U forecastbot_user -W -d m5_db -c "SELECT * FROM m5_sales_train LIMIT 1;"
psql -h localhost -p 5432 -U forecastbot_user -W -d hcmcbus_db -c "SELECT * FROM daily_passenger_stats LIMIT 1;"
psql -h localhost -p 5432 -U forecastbot_user -W -d hcmcbus_db -c "SELECT * FROM bus_routes LIMIT 1;"
```

The metadata should include active entries for `M5` and `HCMC_BUS`. Each business-data query should return a row. These are basic checks, not a complete source-to-target comparison.

## 5. Install dependencies and configure `.env`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

Edit these entries in `.env`, replacing `YOUR_PASSWORD` with the password you set:

```dotenv
POSTGRES_URI=postgresql+psycopg2://forecastbot_user:YOUR_PASSWORD@localhost:5432/forecastbot
FORECASTBOT_DB_URI=postgresql+psycopg2://forecastbot_user:YOUR_PASSWORD@localhost:5432/forecastbot
M5_DB_URI=postgresql+psycopg2://forecastbot_user:YOUR_PASSWORD@localhost:5432/m5_db
HCMC_BUS_DB_URI=postgresql+psycopg2://forecastbot_user:YOUR_PASSWORD@localhost:5432/hcmcbus_db
```

URL-encode special characters in the password when required. These SQLAlchemy URIs are for the application, not for direct use with PostgreSQL CLI tools.

For Gemini, set `GOOGLE_API_KEY`. For the local Gemma model, start Ollama and ensure the model configured by `LOCAL_FORECAST_MODEL` is available. The demo's English translation additionally requires `OPENAI_API_KEY`; a translation failure does not invalidate the forecast.

## 6. Run the demo or an experiment

Start the demo:

```bash
python -m streamlit run app_experiment_chat.py
```

Open the URL printed by Streamlit, select the dataset and model, and submit a question.

To run individual experiment cases:

```bash
python -m evaluation.runner --dataset M5 --model gemini --mode case --case-id M01
python -m evaluation.runner --dataset HCMC_BUS --model gemini --mode case --case-id H01
```

Use `--model gemma` for the configured Ollama model. Live experiment commands can overwrite saved results for the same case, so run new experiments in a separate project copy. See the [experiment guide](../evaluation/README.md) for full runs.
