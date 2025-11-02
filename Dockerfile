# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container (if you have any)
# COPY . /app/

# Set the entrypoint to run your app (if needed)
# ENTRYPOINT ["python", "your_main_script.py"]

# Expose the port (if your app runs on a port)
# EXPOSE 5000

# Optional: Command to run on container start (e.g., testing your setup)
CMD ["python", "--version"]
