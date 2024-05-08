FROM python:3.10.5-slim-bullseye
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt .

#RUN apt-get update
#RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y

RUN pip install -r requirements.txt

COPY . .

# ENTRYPOINT ["python", "app.py"]

# Expose the port your app runs on
EXPOSE 80

# Command to run the Flask app
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "80"]


