# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install frontend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]