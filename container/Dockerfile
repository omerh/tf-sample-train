FROM tensorflow/tensorflow:2.5.0

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl

COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt
RUN pip install sagemaker-training

# ENV SAGEMAKER_PROGRAM train.py
