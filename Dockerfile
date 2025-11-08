# #bas image
# FROM python:3.9

# #workdir
# WORKDIR /app

# #copy
# COPY . /app

# #run
# RUN pip install -r requirements.txt

# #port 
# EXPOSE 8501

# base image
FROM python:3.11-slim

# workdir
WORKDIR /app

# copy requirements first (for better caching)
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application
COPY . .

# port 
EXPOSE 8501

# command - FIXED THIS LINE
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]