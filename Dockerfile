FROM pytorch/pytorch:latest

COPY ./ xlingqg
WORKDIR xlingqg
RUN pip install -r requirements.txt

RUN mkdir models 
WORKDIR models
RUN gdown -O dict.de.txt https://drive.google.com/uc?id=1OrVxjEaYQEPz1tt1ZtC5Fn5VdLjUtKHD
RUN gdown -O dict.en.txt https://drive.google.com/uc?id=1v590QMcRS6HhB397Yv_DbYq1dbA4YK_U
RUN gdown -O bpecodes https://drive.google.com/uc?id=16Da2mkzWYBpMRy72zcxsPILsXoiynulj
RUN gdown -O translation_epoch14.pt https://drive.google.com/uc?id=1Z_SIi6gQz2sCYwnRZzfWjt31HauSxHUI
RUN gdown -O qg_copy_attn_epoch10.pt https://drive.google.com/uc?id=1mIh6ObplggUbto1TMoVuCZNii54CKidX
WORKDIR ..    

EXPOSE 5003

ENTRYPOINT ["python","app.py"]
CMD ["--host", "0.0.0.0", "--port", "5003", "--url_root", ""]