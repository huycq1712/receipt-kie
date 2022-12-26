FROM python:3.7

EXPOSE 8501

COPY . .

RUN apt-get update
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN pip install -r requirements.txt
RUN pip install streamlit
RUN pip install onnx
RUN pip install onnxruntime-gpu

CMD streamlit run app.py