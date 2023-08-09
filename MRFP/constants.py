from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

all_month_oneie_dir = "data/sent_track/allmonth_oneie/"

mftc_dir = "data/MFTC/"
download_dir = "../../../Downloads/"
mftc_data_path = "data/MFTC/MFTC_V4_text.json"
mftc_sandy_dir = f"{mftc_dir}Sandy/"
mftc_all_dir = f"{mftc_dir}All/"
mftc_allfreval_dir = f"{mftc_dir}All_fr_eval/"
mftc_All_fr_eval_dir = f"{mftc_dir}All_fr_eval/"
mftc_fr_path = f"{mftc_allfreval_dir}sampled_twitter_preview.json"
fr_raw_data_path = f"{mftc_allfreval_dir}sampled_twitter_preview.json"

mftc_dir = "data/MFTC/"
labeler_data_dir = "data/response_pred/labeler_data/"
semeval18_dir = f"{labeler_data_dir}semeval18/"
hf_dir = f"{labeler_data_dir}hf/"


TASK_SETTINGS = {
    "response_pred_labeling": 0,
    "response_pred": 1,
    "response_pred_sent": 2,
    "label_sent": 3,
}
TASK_SETTINGS_REVERSE = {v: k for k, v in TASK_SETTINGS.items()}
# LABEL_SPACE={
#     "response_pred_labeling": [0,1,2,3,4,5,6],
#     "response_pred": [0,1,2,3,4,5,6],
#     "response_pred_sent": [0,1,2],
#     "label_sent": [0,1,2],
# }
LABEL_SPACE={
    0: [0,1,2,3,4,5,6],
    1: [0,1,2,3,4,5,6],
    2: [0,1,2],
    3: [0,1,2],
}

LABEL_NAME={
    0: "intensity",
    1: "intensity",
    2: "sentiment",
    3: "sentiment",
}


response_pred_dir = "data/response_pred/"

ALL_STOPWORDS = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat']).Defaults.stop_words
ALL_STOPWORDS.update(set(stopwords.words('english')))
ALL_STOPWORDS.update(set(STOPWORDS))
CNN_dir = f"../twitter_crawl/data_new2/CNN/"
CNN_cgp_dir = f"{CNN_dir}comment_generation_prt"
CNN_raw_file = f"{CNN_dir}all_month_data_no_politics.json"
CNN_raw_tr_file = f"{CNN_dir}all_month_data_no_politics2_emotion_train.json"
CNN_raw_dev_file = f"{CNN_dir}all_month_data_no_politics2_emotion_dev.json"
CNN_raw_te_file = f"{CNN_dir}all_month_data_no_politics2_emotion_test.json"





# moral_label_map = {
#     "non-moral": "nm",
#     "care": "ch",
#     "harm": "ch",
#     "fairness": "fc",
#     "cheating": "fc",
#     "loyalty": "lb",
#     "betrayal": "lb",
#     "authority": "as",
#     "subversion": "as",
#     "purity": "pd",
#     "degradation": "pd",
# }
moral_label_map = {
    "inconfident": -1,
    "authority": 0,
    "betrayal": 1,
    "care": 2,
    "cheating": 3,
    "degradation": 4,
    "fairness": 5,
    "harm": 6,
    "loyalty": 7,
    "non-moral": 8,
    "purity": 9,
    "subversion": 10,
}
# emotion_label_map = {
#     "inconfident": -1,
#     "sadness": 0,
#     "joy": 1,
#     "love": 2,
#     "anger": 3,
#     "fear": 4,
#     "surprise": 5,
# }
emotion_label_map = {
    "inconfident": -1,
    "anger": 0,
    "joy": 1,
    "optimism": 2,
    "sadness": 3,
}
sentiment_label_map = {
    "inconfident": -1,
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

moral_label_map_reverse = {val: key for key, val in moral_label_map.items()}
emotion_label_map_reverse = {val: key for key, val in emotion_label_map.items()}
sentiment_label_map_reverse = {val: key for key, val in sentiment_label_map.items()}

label_map_general = {
    'emotion': emotion_label_map,
    'sentiment': sentiment_label_map,
    'moral': moral_label_map,
}

label_map_general_reverse = {
    'emotion': emotion_label_map_reverse,
    'sentiment': sentiment_label_map_reverse,
    'moral': moral_label_map_reverse,
}