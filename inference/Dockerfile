# Dockerfile for inference
# Start from a base Python image
FROM python:3.10

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy the code itself
COPY inference /app/inference

# Copy utils.py
COPY utils.py /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run the inference script with the default inference data file as an argument
CMD python inference/run.py --infer_file ${INFERENCE_FILE} --out_path ${OUT_PATH}
