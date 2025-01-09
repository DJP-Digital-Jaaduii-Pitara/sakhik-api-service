FROM python:3.8.10
WORKDIR /code
RUN apt-get update && apt install build-essential --fix-missing -y
RUN apt-get install ffmpeg -y
COPY ./requirements-prod.txt /code/requirements-prod.txt
RUN python -m pip install --upgrade pip
ARG CONFIG_FILE_PATH
COPY $CONFIG_FILE_PATH /config/config
COPY . /config
RUN pip install --no-cache-dir --upgrade -r /code/requirements-prod.txt
COPY . /code
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "8"]

# FROM continuumio/anaconda3:2023.03-1
# WORKDIR /root
# RUN apt-get update && apt-get install -y curl file
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH=$PATH:/root/.cargo/bin \
#     OPENAI_API_KEY=$OPENAI_API_KEY \
#     LOG_LEVEL=$LOG_LEVEL \
#     BHASHINI_ENDPOINT_URL=$BHASHINI_ENDPOINT_URL \
#     BHASHINI_API_KEY=$BHASHINI_API_KEY \
#     OCI_ENDPOINT_URL=$OCI_ENDPOINT_URL \
#     OCI_REGION_NAME=$OCI_REGION_NAME \
#     OCI_BUCKET_NAME=$OCI_BUCKET_NAME \
#     OCI_SECRET_ACCESS_KEY=$OCI_SECRET_ACCESS_KEY \
#     OCI_ACCESS_KEY_ID=$OCI_ACCESS_KEY_ID \
#     MARQO_URL=$MARQO_URL \
#     SERVICE_ENVIRONMENT=$SERVICE_ENVIRONMENT \
#     TELEMETRY_ENDPOINT_URL=$TELEMETRY_ENDPOINT_URL \
#     TELEMETRY_LOG_ENABLED=$TELEMETRY_LOG_ENABLED
# RUN apt-get update && apt install build-essential --fix-missing -y
# RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.05.tar.gz &&  \
#     tar -xvf xpdf-tools-linux-4.05.tar.gz && cp xpdf-tools-linux-4.05/bin64/pdftotext /usr/local/bin
# RUN apt-get install ffmpeg -y
# COPY requirements-prod.txt /root/
# RUN pip3 install -r requirements-prod.txt
# COPY ./main.py /root/
# COPY ./query_with_langchain.py /root/
# COPY ./io_processing.py /root/
# COPY ./logger.py /root/
# COPY ./utils.py /root/
# COPY ./telemetry_logger.py /root/
# COPY ./telemetry_middleware.py /root/
# COPY ./config.ini /root/
# COPY ./config_util.py /root/
# COPY ./env_manager.py /root/
# COPY ./llm /root/
# COPY ./storage /root/
# COPY ./translation /root/
# EXPOSE 8000
# COPY script.sh /root/
# ENTRYPOINT ["bash","script.sh"]
