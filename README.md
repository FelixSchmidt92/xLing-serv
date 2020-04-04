# xLingQG-serv
Cross lingual question generation server.
Generates reading comprehension questions in german from english texts.

# Getting started
## Requirements: 
- python 3.6

## Setup
Setup and activate virtual environment
````
virtualenv -p python3 venv
source venv/bin/activate
````

Install requirements
````
git clone https://github.com/FelixSchmidt92/xLingQG-serv.git
cd xlingQG-serv
pip install -r requirements.txt
python -m spacy download en_core_web_lg
````
Neuralcoref has to be installed from source in order to work with spacy
````
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
````

## Download models
Download the pretrained models for question generation and translation
````
mkdir models 
cd models
gdown -O dict.de.txt https://drive.google.com/uc?id=1OrVxjEaYQEPz1tt1ZtC5Fn5VdLjUtKHD
gdown -O dict.en.txt https://drive.google.com/uc?id=1v590QMcRS6HhB397Yv_DbYq1dbA4YK_U
gdown -O bpecodes https://drive.google.com/uc?id=16Da2mkzWYBpMRy72zcxsPILsXoiynulj
gdown -O translation_epoch14.pt https://drive.google.com/uc?id=1Z_SIi6gQz2sCYwnRZzfWjt31HauSxHUI
gdown -O qg_copy_attn_epoch10.pt https://drive.google.com/uc?id=1mIh6ObplggUbto1TMoVuCZNii54CKidX
cd ..    
````

## Generate questions
start the server on localhost port 5000

````
python app.py --host 0.0.0.0 --port 5000
````

Send request to the server to generate questions
Request:
````
curl --request POST \
  --data '{"text":"Roger Federer was born in Switzerland in 1981. He has won 20 titles. He is a tennis player."}' \
  http://localhost:5000/Question
````
Response:
````
[{"answer":"Roger Federer","question":"Wer wurde 1981 in der Schweiz geboren?","sentence":"Roger Federer was born in Switzerland in 1981."},{"answer":"in der Schweiz","question":"Wo wurde Roger Federer geboren?","sentence":"Roger Federer was born in Switzerland in 1981."},{"answer":"1981","question":"Wann wurde Roger Federer geboren?","sentence":"Roger Federer was born in Switzerland in 1981."},{"answer":"20 Titel","question":"Wie viele Titel hat Roger Federer gewonnen?","sentence":"Roger Federer has won 20 titles."},{"answer":"Roger Federer","question":"Wer hat 20 Titel gewonnen?","sentence":"Roger Federer has won 20 titles."},{"answer":"ein Tennisspieler","question":"Welche Art von Spieler ist Roger Federer?","sentence":"Roger Federer is a tennis player."},{"answer":"Roger Federer","question":"Wer ist Tennisspieler?","sentence":"Roger Federer is a tennis player."}]
````

## Generate on GPU
In order to genereate on GPU change the properties gpu in modelconfig/config.json for translation model to true and for question generation model to 0


# build image and run 
Increase the maximum Memory docker can consume in the settings of docker on your local machine. 
The docker image does currently not support inference on gpu.
````
docker build -t xqg .
docker run -p 5000:5000 xqg
````

# Translation and Question Generation Training
- Translation Model Training: https://github.com/FelixSchmidt92/nmt_training
- QG-Model Training: https://github.com/FelixSchmidt92/qg_training

# evaluation
For the evaluation of the pretrained models the colab notebook in this repository can be used.  
