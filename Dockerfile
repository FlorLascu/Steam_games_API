FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file from the host to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the entire project directory into the docker container 
COPY . .

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]