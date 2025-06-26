#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Start the FastAPI app (adjust as needed)
exec uvicorn main:app --host 0.0.0.0 --port 10000
