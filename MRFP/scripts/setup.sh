mkdir data
mkdir data/response_pred
mkdir data/response_pred/labeler_data

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m" -O BARTScore/bart.pth && rm -rf /tmp/cookies.txt


## torch
chmod +x scripts/*
python -m pip install -r requirements.txt
python -m spacy download en
python -c "import nltk; nltk.download('stopwords', quiet=True)"
#python -c "import wandb; wandb.login(key='')"

