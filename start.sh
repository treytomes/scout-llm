echo "Starting web server..."
source .venv/bin/activate
uvicorn app:app --app-dir ./src/server --root-path ./src --reload
