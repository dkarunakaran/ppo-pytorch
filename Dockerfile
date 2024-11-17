FROM python:3.12.5-bookworm

RUN pip install --upgrade pip
##### Core scientific packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install matplotlib~=3.8.1
RUN pip install numpy~=1.26.2
RUN pip install pandas~=2.1.3
RUN pip install gymnasium
RUN pip install PyYAML

WORKDIR /app

CMD ["/bin/bash"]