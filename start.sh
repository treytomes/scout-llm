echo "Starting web server..."
source .venv/bin/activate
uvicorn server.app:app --app-dir ./src --root-path ./src --reload
