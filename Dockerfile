# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . /app/

# Default command to run when the container starts
CMD ["python3", "standardize.py", "--nucc", "nucc_taxonomy_master.csv", "--input", "input_specialties.csv", "--synonyms", "synonyms.csv", "--out", "output.csv"]
