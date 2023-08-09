import os
from multiprocessing import Pool
# import multiprocessing as mp
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string
from utils.utils import load_file, dump_file, get_directory, path_exists, module_exists
# print("imported utils")
from IPython import embed
from utils.graph import load_resources
from collections import OrderedDict
from pprint import pprint as pp
import re

# from flair.data import Sentence
# from flair.models import SequenceTagger
if module_exists("transition_amr_parser"):
    from transition_amr_parser.stack_transformer_amr_parser import AMRParser
from process_amr import amr_parse, processing_amr
__all__ = ['create_matcher_patterns', 'ground']

# the lemma of it/them/mine/.. is -PRON-

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])

# nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

# CHUNK_SIZE = 1

CPNET_VOCAB = None
PATTERN_PATH = None
nlp = None
matcher = None
cur_concept2id = None
cur_id2concept = None
record_dict={}
pos_tagger= None


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_pattern(nlp, doc, debug=False):
    pronoun_list = set(
        ["my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our", "we"])
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or \
            all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords or token.lemma_ in blacklist) for token
                 in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def create_matcher_patterns(cpnet_vocab_path, output_path, debug=False):
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    if debug:
        f = open("filtered_concept.txt", "w")

    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc, debug)
        if debug:
            if not pattern[0]:
                f.write(pattern[1] + '\n')

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()


# def lemmatize(nlp, concept):
#     doc = nlp(concept.replace("_", " "))
#     lcs = set()
#     # for i in range(len(doc)):
#     #     lemmas = []
#     #     for j, token in enumerate(doc):
#     #         if j == i:
#     #             lemmas.append(token.lemma_)
#     #         else:
#     #             lemmas.append(token.text)
#     #     lc = "_".join(lemmas)
#     #     lcs.add(lc)
#     lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
#     return lcs


def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, None, pattern)
    return matcher


def ground_sent(s):  # s in a tuple
    # global nlp, matcher

    global nlp, matcher

    # print("\nground_sent")
    if nlp is None or matcher is None:
        # print("PATTERN_PATH", PATTERN_PATH)
        # print("CPNET_VOCAB", type(CPNET_VOCAB))

        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = load_matcher(nlp, PATTERN_PATH)

    global cur_concept2id
    # if any(x is Ncone for x in [cur_concept2id, cur_id2concept]):
    #     # print("load_resources")
    #     with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
    #         cur_id2conept = [w.strip() for w in fin]
    #     cur_concept2id = {w: i for i, w in enumerate(cur_id2concept)}

    # print("\n\n", s, "\n\n")
    question_concepts, concept_id_to_pos_list,  doc = ground_mentioned_concepts(nlp, matcher, s, cpnet_vocab=CPNET_VOCAB)
    # if "being yourself" in question_concepts:
    #     breakpoint()
    # for c in question_concepts:
    #     if c not in cur_concept2id:
    #         embed()

    assert all([cur_concept2id[c] in concept_id_to_pos_list for c in question_concepts]), breakpoint()
    # print("question_concepts", question_concepts)
    # question_concepts, span_to_final_concepts, pos_to_conceptids, doc = ground_mentioned_concepts(nlp, matcher, s)
    # if len(question_concepts) == 0:
    #     question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible
    question_concepts = sorted(list(question_concepts))
    # print("q concepts updated hard_ground", question_concepts)


    return {"sent": s, "ans": "", "qc": question_concepts, "ac": [], "concept_id_to_pos_list":concept_id_to_pos_list, "doc":doc}
    return {"sent": s, "ans": "", "qc": question_concepts, "ac": [], "span_to_final_concepts":span_to_final_concepts, "pos_to_conceptids":pos_to_conceptids, "doc":doc}

    # s, nlp, matcher, CPNET_VOCAB = s
    #
    # print("s", s)
    # # global nlp, matcher, PATTERN_PATH
    # # print("PATTERN_PATH", PATTERN_PATH)
    # # if nlp is None or matcher is None:
    # #     nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    # #     nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # #     matcher = load_matcher(nlp, PATTERN_PATH)
    # print("nlp", nlp, type(nlp))
    # print("matcher", matcher, type(matcher))
    # print("PATTERN_PATH", PATTERN_PATH)
def ground_qa_pair(qa_pair):
    global nlp, matcher
    if nlp is None or matcher is None:
        print("PATTERN_PATH", PATTERN_PATH)
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = load_matcher(nlp, PATTERN_PATH)

    s, a = qa_pair
    all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
    answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
    question_concepts = all_concepts - answer_concepts
    if len(question_concepts) == 0:
        question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible

    if len(answer_concepts) == 0:
        answer_concepts = hard_ground(nlp, a, CPNET_VOCAB)  # some case

    # question_concepts = question_concepts -  answer_concepts
    question_concepts = sorted(list(question_concepts))
    answer_concepts = sorted(list(answer_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts, "ac": answer_concepts}





def ground_mentioned_concepts(nlp, matcher, s, ans=None, cpnet_vocab=None):

    global cur_concept2id

    # doc_case_sensitive = nlp(s) # tokenized
    # s = s.lower()
    doc = nlp(s.lower()) # tokenized

    # embed()
    matches = matcher(doc)
    mentioned_concepts = set()
    span_to_concepts = {}
    span_to_final_concepts = {}
    # pos_to_concepts = {}
    # pos_to_spans = {}
    # pos_to_conceptids = {}

    span_to_pos = {}
    concept_id_to_pos_list = {}
    # print("doc",doc)
    # print("\ndoc_case_sensitive",list(doc_case_sensitive))
    # print("doc",list(doc))
    # print("doc",doc)
    # print("list(doc)",  list(doc))
    # print("list(doc_case_sensitive)",  list(doc_case_sensitive))


    org_tokens=[str(itm) for itm in list(doc)]
    sensti_tokens=[s[token_info['start']:token_info['end']] for token_info in doc.to_json()['tokens']] # matching needs lower case, so recover upper
    # sensti_tokens=[str(itm) for itm in list(doc_case_sensitive)]
    # if org_tokens[0]=="paint" and org_tokens[-1]=="roller":
    #     embed()
    # print(org_tokens, sensti_tokens)
    assert len(list(sensti_tokens))==len(list(doc)), print("not equal length", org_tokens, sensti_tokens)
    for i, tkn in enumerate(org_tokens):
        assert sensti_tokens[i].lower()==tkn, print("not match token length",org_tokens, sensti_tokens)

    # print("matches", matches)

    if ans is not None:
        ans_matcher = Matcher(nlp.vocab)
        ans_words = nlp(ans)
        # print(ans_words)
        ans_matcher.add(ans, None, [{'TEXT': token.text.lower()} for token in ans_words])

        ans_match = ans_matcher(doc)
        ans_mentions = set()
        for _, ans_start, ans_end in ans_match:
            ans_mentions.add((ans_start, ans_end))

    for match_id, start, end in matches:
        if ans is not None:
            if (start, end) in ans_mentions:
                continue

        # print("start, end",start, end)
        # print(doc[start:end])
        assert start<end, embed()



        span = doc[start:end].text  # the matched span

        # a word that appears in answer is not considered as a mention in the question
        # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
        #     continue
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = set()
        original_concept_set.add(original_concept)

        # print("original_concept", original_concept)

        # print("span", span)
        # print("concept", original_concept)
        # print("Matched '" + span + "' to the rule '" + string_id)

        # why do you lemmatize a mention whose len == 1?
        if len(original_concept.split("_")) == 1:
            # tag = doc[start].tag_
            # if tag in ['VBN', 'VBG']:

            original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))
        # print("original_concept_set", original_concept_set)
        ##maybe same word differnt pos at dif pos, so lead to different linked concepts?

        if span not in span_to_concepts:
            span_to_concepts[span] = set()
        span_to_concepts[span].update(original_concept_set)
        # if frozenset((start, end)) not in pos_to_concepts:
        #     pos_to_concepts[frozenset((start, end))] = set()
        # if frozenset((start, end)) not in pos_to_spans:
        #     pos_to_spans[frozenset((start, end))] = set()
        # pos_to_spans[frozenset((start, end))].add(span)
        # pos_to_concepts[frozenset((start, end))].update(original_concept_set)

        if span not in span_to_pos:
            span_to_pos[span] = set()
        span_to_pos[span].add((start, end))

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        # print("span:")
        # print(span)
        # print("concept_sorted:")
        # print(concepts_sorted)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3]

        if span not in span_to_final_concepts:
            span_to_final_concepts[span]=set()
        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
                span_to_final_concepts[span].add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)
                span_to_final_concepts[span].add(c)


        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
        # print("exact match:")
        # print(exact_match)
        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)
        span_to_final_concepts[span].update(exact_match)

        # cur_ccpts=span_to_final_concepts[span]
        # print("sensti_tokens", sensti_tokens, "cur_ccpts", cur_ccpts, "span", span)
        cur_ccptsid=[cur_concept2id[ccpt] for ccpt in span_to_final_concepts[span] if ccpt in cur_concept2id]

        for ccptid in cur_ccptsid: # if ccpt in cur_concept2id
            if ccptid not in concept_id_to_pos_list:
                concept_id_to_pos_list[ccptid]=set()
            concept_id_to_pos_list[ccptid].update(span_to_pos[span])
    # remove those not in vocab
    mentioned_concepts=set([item for item in mentioned_concepts if item in cur_concept2id])


    # for pos, spans in pos_to_spans.items():
    #     if pos not in pos_to_conceptids:
    #         pos_to_conceptids[pos]=set()
    #     for span in spans:
    #         pos_to_conceptids[pos].update([cur_concept2id[ccpt] for ccpt in span_to_final_concepts[span]])
    # print("\n\ndoc",list(doc))
    # print("pos_to_conceptids", pos_to_conceptids)
    # print("\n\nspan_to_pos", span_to_pos)
    # print("span_to_final_concepts", span_to_final_concepts)
    # print("mentioned_concepts", mentioned_concepts)
    # print("concept_id_to_pos_list", concept_id_to_pos_list)
    # embed()
    # print([str(itm) for itm in list(doc_case_sensitive)])

    ##hard ground
    if not len(mentioned_concepts):
        print("not matching any, so hard grounding...")
        assert not concept_id_to_pos_list, breakpoint()
        concept_id_to_pos_list={}
        for i, t in enumerate(doc):
            if t.lemma_ in cur_concept2id:
                mentioned_concepts.add(t.lemma_)
                cur_ccpt_id=cur_concept2id[t.lemma_]
                if cur_ccpt_id not in concept_id_to_pos_list:
                   concept_id_to_pos_list[cur_ccpt_id]=set()
                concept_id_to_pos_list[cur_ccpt_id].add((i,i+1))
        sent = "_".join([t.text for t in doc]).strip()
        if sent in cur_concept2id:
            mentioned_concepts.add(sent)
            cur_ccpt_id=cur_concept2id[sent]
            if cur_ccpt_id not in concept_id_to_pos_list:
               concept_id_to_pos_list[cur_ccpt_id]=set()
            concept_id_to_pos_list[cur_ccpt_id].add((0, len(doc)))
        try:
            assert len(mentioned_concepts) > 0
        except Exception:
            print(f"for {sent}, concept not found in hard grounding.")

        print("doc", list(doc))
        print("mentioned_concepts", mentioned_concepts)
        print("concept_id_to_pos_list", concept_id_to_pos_list)

    for ccid in concept_id_to_pos_list:
        concept_id_to_pos_list[ccid]=list(concept_id_to_pos_list[ccid])#sorted()
    # print("\n\nsensti_tokens", sensti_tokens,"\nspan_to_final_concepts", span_to_final_concepts)

    return mentioned_concepts, concept_id_to_pos_list, sensti_tokens #sensti_tokens #doc.to_json()
    return mentioned_concepts, span_to_final_concepts, pos_to_conceptids, doc


def hard_ground(nlp, sent, cpnet_vocab):
    global cur_concept2id

    concept_id_to_pos_list={}
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for i,t in enumerate(doc):
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    try:
        assert len(res) > 0
    except Exception:
        print(f"for {sent}, concept not found in hard grounding.")
    return res


def match_mentioned_concepts(sents, answers, num_processes, sent_only=True):
    print("match_mentioned_concepts")

    if sent_only:
        if num_processes<=1:
            res=[ground_sent(sent) for sent in tqdm(sents)]
        else:
            with Pool(num_processes) as p:
                res = list(tqdm(p.imap(ground_sent, sents), total=len(sents)))
    else:
        if num_processes<=1:
            res=[ground_qa_pair((sent, ans)) for sent, ans in zip(sents, answers)]
        else:
            with Pool(num_processes) as p:
                res = list(tqdm(p.imap(ground_qa_pair, zip(sents, answers)), total=len(sents)))
    return res

def parse_sent(s):

    global nlp, matcher
    # print("\nground_sent")
    if nlp is None or matcher is None:
        # print("PATTERN_PATH", PATTERN_PATH)
        # print("CPNET_VOCAB", type(CPNET_VOCAB))

        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # matcher = load_matcher(nlp, PATTERN_PATH)

    # embed()
    return [str(ss) for ss in list(nlp(s))]
    return [s[token_info['start']:token_info['end']]
                    for token_info in nlp(s).to_json()['tokens']]

def parse_into_tokens(sents, num_processes):


    global nlp, matcher
    # print("\nground_sent")
    if nlp is None or matcher is None:
        # print("PATTERN_PATH", PATTERN_PATH)
        # print("CPNET_VOCAB", type(CPNET_VOCAB))

        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # matcher = load_matcher(nlp, PATTERN_PATH)

    prep_d=[(sent) for sent in sents]

    if num_processes<=1:
        res=[parse_sent((sent)) for sent in tqdm(sents)]
    else:
        with Pool(num_processes) as p:
            res = list(tqdm(p.imap(parse_sent, prep_d), total=len(sents)))
    return res

# To-do: examine prune
def prune(data, cpnet_vocab_path, no_ans=True):
    print("prune")
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    global cur_concept2id, cur_id2concept
    if any(x is None for x in [cur_concept2id, cur_id2concept]):
        # print("load_resources")
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            cur_id2concept = [w.strip() for w in fin]
        cur_concept2id = {w: i for i, w in enumerate(cur_id2concept)}

    prune_data = []
    for item in tqdm(data):
        qc = item["qc"]
        concept_id_to_pos_list = item["concept_id_to_pos_list"]
        prune_qc = []
        prune_concept_id_to_pos_list={}
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune_qc.append(c)
                prune_concept_id_to_pos_list[cur_concept2id[c]]=concept_id_to_pos_list[cur_concept2id[c]]
        if not no_ans:
            ac = item["ac"]
            concept_id_to_pos_list = item["concept_id_to_pos_list"]
            prune_ac = []
            prune_concept_id_to_pos_list={}
            for c in ac:
                if c[-2:] == "er" and c[:-2] in ac:
                    continue
                if c[-1:] == "e" and c[:-1] in ac:
                    continue
                all_stop = True
                for t in c.split("_"):
                    if t not in nltk_stopwords:
                        all_stop = False
                if not all_stop and c in cpnet_vocab:
                    prune_ac.append(c)
                    prune_concept_id_to_pos_list[cur_concept2id[c]]=concept_id_to_pos_list[cur_concept2id[c]]

        try:
            assert len(prune_qc) > 0
            if not no_ans: assert len(prune_ac) > 0
        except Exception as e:
            pass
            # print("In pruning")
            # print("prune_qc", prune_qc)
            # print(prune_ac)
            # print("original:")
            # print(qc)
            # print(ac)
            # print()
        item["qc"] = prune_qc
        item["concept_id_to_pos_list"] = prune_concept_id_to_pos_list
        if not no_ans:
            item["ac"] = prune_ac

        prune_data.append(item)
    return prune_data


def ground(statement_path, cpnet_vocab_path, pattern_path, output_path, num_processes=1, debug=False):
    global PATTERN_PATH, CPNET_VOCAB
    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)

    global cur_concept2id, id2concept
    if any(x is None for x in [cur_concept2id, id2concept]):
        # print("load_resources")
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            id2concept = [w.strip() for w in fin]
        cur_concept2id = {w: i for i, w in enumerate(id2concept)}

    sents = []
    answers = []
    with open(statement_path, 'r') as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[192:195]
        print(len(lines))
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        # {'answerKey': 'B',
        #   'id': 'b8c0a4703079cf661d7261a60a1bcbff',
        #   'question': {'question_concept': 'magazines',
        #                 'choices': [{'label': 'A', 'text': 'doctor'}, {'label': 'B', 'text': 'bookstore'}, {'label': 'C', 'text': 'market'}, {'label': 'D', 'text': 'train station'}, {'label': 'E', 'text': 'mortuary'}],
        #                 'stem': 'Where would you find magazines along side many other printed works?'},
        #   'statements': [{'label': False, 'statement': 'Doctor would you find magazines along side many other printed works.'}, {'label': True, 'statement': 'Bookstore would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Market would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Train station would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Mortuary would you find magazines along side many other printed works.'}]}

        for statement in j["statements"]:
            sents.append(statement["statement"])

        for answer in j["question"]["choices"]:
            ans = answer['text']
            # ans = " ".join(answer['text'].split("_"))
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

    res = match_mentioned_concepts(sents, answers, num_processes, sent_only=False)
    res = prune(res, cpnet_vocab_path)

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()



def tokenize(paragraph,nlp,nlpp=None):
    """
    Change the paragraph to lower case and tokenize it!
    """
    # paragraph = re.sub(' +', ' ', paragraph)  # remove redundant spaces in some sentences.
    # para_doc = nlp(paragraph)  # create a SpaCy Doc instance for paragraph

    para_doc = nlp(paragraph.lower())  # create a SpaCy Doc instance for paragraph
    alt=nlp(paragraph)
    if [tok.text.lower() for tok in alt]!=[tok.text for tok in para_doc]:
        para_doc = alt
    # aa=nlp(step)
    # bb=nlp(step.lower())
    # assert [tok.text.lower() for tok in aa]==[tok.text for tok in bb]

    # para_doc = nlp(paragraph)  # create a SpaCy Doc instance for paragraph

    # a=nlpp(paragraph.lower())
    # if len(a)!=len(para_doc):
    #     embed()

    # tokens_list = [token.text for token in para_doc]
    # tokens_list = [token for token in para_doc]
    return para_doc
    return ' '.join(tokens_list), len(tokens_list), tokens_list


def lemmatize(paragraph: str,nlp):
    """
    Reads a paragraph/sentence/phrase/word and lemmatize it!
    """
    if paragraph == '-' or paragraph == '?':
        return None, paragraph
    # a=paragraph.split()
    para_doc = nlp(paragraph)
    # b=[token.text  for token in para_doc]
    # assert a==b

    lemma_list = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in para_doc]
    return lemma_list, ' '.join(lemma_list)
def find_vb_nn(paragraph,nlp, orig_len=None,orig_tokens=None, record_dict=None):#pos_tagger,
    """
    paragraph: the paragraph after tokenization and lower-case transformation
    return: the location candidates found in this paragraph
    """
    # pos_tagger.predict(paragraph)
    # pos_list = [(token.text, token.get_tag('pos').value) for token in paragraph]
    pos_list = [(token.text, token.tag_) for token in paragraph]

    # print('len(pos_list)',len(pos_list))
    # assert len(pos_list)==orig_len

    loc_list = []
    vb_list = []
    nns=[]
    vb_idxs=[]
    # coref_dict=defaultdict(list)
    coref_dict={}
    cur_v_idx=(-1,-1)
    cur_v=None

    exclude_set={"minute", "minutes","hours", "hour","seconds"}
    # extract nouns (including 'noun + noun' phrases)
    for i in range(len(pos_list)):
        if pos_list[i][0].lower() not in exclude_set  \
                and 'NN' in pos_list[i][1] and (i==len(pos_list)-1 or 'NN' not in pos_list[i+1][1]):
            candidate = pos_list[i][0]
            # candidate = [pos_list[i][0]]


            s,e=i,i+1
            for k in range(1, i + 1): #'JJ' in pos_list[i - k][1] or
                if 'NN' in pos_list[i - k][1]:# in ['ADJ','NN','NNS',"NNP","NNPS"]: or 'HYPH' in pos_list[i - k][1]  or 'CD' in pos_list[i - k][1] # coreferece can't do CD!
                    s=i - k
                    candidate = pos_list[i - k][0] + ' ' + candidate
                    # candidate = [pos_list[i - k][0]]  + candidate
                else:
                    break
            if candidate not in record_dict:
                _, lemma = lemmatize(candidate, nlp)
                record_dict[candidate.lower()]=lemma.lower()
            res = [[s, e]]
            if -1 not in cur_v_idx:
                res.append(cur_v_idx)
            coref_dict[record_dict[candidate.lower()]]=res#list(cur_v_idx), [s,e]

            # _, lemma = lemmatize(candidate,nlp)
            # # coref_dict[lemma].append((cur_v_idx, (s,e)))
            # coref_dict[lemma]=((s,e),cur_v_idx)#list(cur_v_idx), [s,e]
        elif 'VB' in pos_list[i][1]: # in ['VB', "VBG", "VBD", "VB", "VB", "VB", "VB", "VB"]
            # print("VB",pos_list[i][0])
            # candidate = pos_list[i][0]
            s,e=i,i+1
            # print(s,e)
            # _, lemma = lemmatize(pos_list[i][0])
            # vb_idxs.append((s,e))
            cur_v_idx=[s,e]
            cur_v=pos_list[i][0]

    # new_coref_dict={}
    # for key,item in coref_dict.items():
    #     _, lemma = lemmatize(key, nlp)
    #     new_coref_dict[lemma] = item  # list(cur_v_idx), [s,e]
    # coref_dict=new_coref_dict
    # for candidate:
    #     _, lemma = lemmatize(candidate, nlp)

    return coref_dict

    # # # extract 'noun + and/or + noun' phrase
    # # for i in range(2, len(pos_list)):
    # #     if pos_list[i][1] == 'NN' \
    # #             and (pos_list[i - 1][0] == 'and' or pos_list[i - 1][0] == 'or') \
    # #             and pos_list[i - 2][1] == 'NN':
    # #         loc_list.append(pos_list[i - 2][0] + ' ' + pos_list[i - 1][0] + ' ' + pos_list[i][0])
    # #
    # # # noun + of + noun phrase
    # # for i in range(2, len(pos_list)):
    # #     if pos_list[i][1] == 'NN' \
    # #             and pos_list[i - 1][0] == 'of' \
    # #             and pos_list[i - 2][1] == 'NN':
    # #         loc_list.append(pos_list[i - 2][0] + ' ' + pos_list[i - 1][0] + ' ' + pos_list[i][0])
    # #
    # # # noun + of + a/an/the + noun phrase
    # # for i in range(3, len(pos_list)):
    # #     if pos_list[i][1] == 'NN' \
    # #             and pos_list[i - 1][1] == 'DT' \
    # #             and pos_list[i - 2][0] == 'of' \
    # #             and pos_list[i - 3][1] == 'NN':
    # #         loc_list.append(pos_list[i - 3][0] + ' ' + pos_list[i - 2][0] + ' ' + pos_list[i - 1][0] + ' ' + pos_list[i][0])
    #
    # # lemmatization
    # coref_dict={}
    # for i in range(len(loc_list)):
    #     _, location = lemmatize(loc_list[i])
    #     loc_list[i] = location
    #
    # return set(loc_list)

def extract_vn_sent(d):

    global nlp, record_dict

    # print("\nground_sent")
    if nlp is None or not record_dict: #or pos_tagger is None
        # print("PATTERN_PATH", PATTERN_PATH)
        # print("CPNET_VOCAB", type(CPNET_VOCAB))

        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # matcher = load_matcher(nlp, PATTERN_PATH)
        # pos_tagger=  SequenceTagger.load('pos')
        record_dict={}

    step=d

    # paragraph, total_tokens, tokens_list = tokenize(step, nlp)
    tokenized = tokenize(step, nlp)
    coref_dict = find_vb_nn(tokenized, nlp, record_dict=record_dict)

    # tokenized = Sentence(paragraph)
    # coref_dict = find_vb_nn(tokenized, pos_tagger, nlp, record_dict=record_dict)
    return [tok.text for tok in tokenized],coref_dict#,vb_idxs
    # sample.update({
    #     "tokens": [tok.text for tok in tokenized.tokens],
    #     "coref_dict": coref_dict,
    #     "vb_idxs": vb_idxs,
    # })


# noinspection PyTypeChecker
def ground_sents(statement_path, cpnet_vocab_path, pattern_path, output_path, num_processes=1, collect_ccpts=False,
                 batch_size=256, roberta_batch_size=45, full_seq_only=False, do_amr=False, do_est=True):
    print("output_path",output_path)
    print("bbb", batch_size, roberta_batch_size)
    print("full_seq_only",full_seq_only)
    print("do_amr",do_amr)
    print("do_est",do_est)
    global PATTERN_PATH, CPNET_VOCAB
    if collect_ccpts:
        if PATTERN_PATH is None:
            PATTERN_PATH = pattern_path
            CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)

        global cur_concept2id, cur_id2concept
        if any(x is None for x in [cur_concept2id, cur_id2concept]):
            # print("load_resources")
            with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
                cur_id2concept = [w.strip() for w in fin]
            cur_concept2id = {w: i for i, w in enumerate(cur_id2concept)}



    global nlp, matcher, record_dict, pos_tagger
    # print("\nground_sent")
    if nlp is None or matcher is None :# or pos_tagger is None
        # print("PATTERN_PATH", PATTERN_PATH)
        # print("CPNET_VOCAB", type(CPNET_VOCAB))

        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # matcher = load_matcher(nlp, PATTERN_PATH)
        # pos_tagger = SequenceTagger.load('pos')
        record_dict = {}

    samples=load_file(statement_path)

    # Input src text is list of steps
    # sents = [" [SEP] ".join(item["src_text"]) for item in samples]
    s_pos=0
    # for i in range(len(samples)):
    #     if " ".join(['Mix', 'a', 'solution', 'of', 'warm', 'water']) in samples[i]["step"]:
    #         s_pos=i
    #         print("found", i)
    #         break

    sents = [item["step"] for item in samples]


    """====parse sents to tokens lists===="""
    # embed()
    if do_amr:
        for suffix in ["_train","_dev","_test","_full"]:
            if suffix in output_path:
                cur_path=os.path.join(get_directory(output_path), f"tokens_lists{suffix}.json")
                if path_exists(cur_path):
                    tokens_lists = load_file(cur_path)
                else:
                    # tokens_lists_cls=[Sentence(sent) for sent in sents]
                    # tokens_lists=[tok.text for sent in tokens_lists_cls for tok in sent.tokens ]
                    tokens_lists = [[sents[i][token_info['start']:token_info['end']]
                                     for token_info in tokenized_s.to_json()['tokens']] for i, tokenized_s in enumerate(
                        nlp.pipe(sents, batch_size=10000))]  # matching needs lower case, so recover upper
                    dump_file(tokens_lists, cur_path)

                break
        for i,tk in enumerate(tokens_lists):
            tokens_lists[i]=tk[:50]

    # embed()
    # tokens_lists = nlp.pipe(sents, batch_size=50)
    #
    # tokens_lists = parse_into_tokens(sents, num_processes)

    # print("e1")
    #
    #
    # print(tokens_lists[0], sents[0])
    # print(tokens_lists[1], sents[1])
    #
    # print('tokens_lists')
    # breakpoint()

    """get lemma"""
    # ctx = mp.get_context("spawn")
    if do_est:
        if num_processes <= 1:
            res = [extract_vn_sent((step)) for step in sents]
        else:
            with Pool(num_processes) as p:#ctx.
                res = list(tqdm(p.imap(extract_vn_sent, sents), total=len(sents)))

        for i, sample in enumerate(samples):
           sample['new_doc']={
                    "tokens":res[i][0],
                    "coref_dict":res[i][1],
                    # "vb_idxs":res[i][2],
                    }
        # embed()
       # sample.update({
       #          "tokens":res[i][0],
       #          "coref_dict":res[i][1],
       #          # "vb_idxs":res[i][2],
       #          })
        # sample.update({
        #         "tokens":[tok.text for tok in tokenized.tokens],
        #         "coref_dict":coref_dict,
        #         "vb_idxs":vb_idxs,
        #         })
    # embed()
    # breakpoint()

    answers = ["" for _ in samples] # Dummy var, won't be used
    # print("len(sents)", len(sents))

    """====if need to link to concepts===="""
    if collect_ccpts:
        res = match_mentioned_concepts(sents, answers, num_processes, sent_only=True)
        dump_file(res, output_path)
        res = prune(res, cpnet_vocab_path)

    if not full_seq_only and do_amr:
        """====get amr graphs===="""
        parser = AMRParser.from_checkpoint('../../transition-amr-parser/amr_general/checkpoint_best.pt')
        # parser = AMRParser.from_checkpoint('/home/chenkai5/transition-amr-parser/amr_general/checkpoint_best.pt')
        # parser = AMRParser.from_checkpoint('/root/transition-amr-parser/amr_general/checkpoint_best.pt')


        bsz=1000
        start_pos=0
        amr_graphs, align, exist, amr_list = [], [], [], []
        for suffix in ["_train","_dev","_test"]:
            if suffix in output_path:
                cur_path=os.path.join(get_directory(output_path), f"amr{suffix}.json")
                if path_exists(cur_path):
                    amr_graphs, align, exist, amr_list = load_file(cur_path)
                    start_pos=len(amr_graphs)
                break

        print("start_pos",start_pos)

        for j in range(start_pos, len(tokens_lists), bsz):
            print("grounding j",j)

            """cut of over long sents"""
            for ind in range(j,min(j+bsz, len(tokens_lists))):
                # if "smallurl" in tokens_lists[ind]:
                # print(tokens_lists[ind])
                tokens_lists[ind]=tokens_lists[ind][:50]
                for k, tk in enumerate(tokens_lists[ind]):
                    # if "smallUrl" in tk or "/" in tk:
                    #     # t = re.sub(r"\{\"smallUrl\".+\}", '', tk)
                    #     # if t:
                    #     #     t = re.sub(r"https?://(www\.)?", '', t)
                    #     #
                    #     # # HTML
                    #     # if t:
                    #     #     t = re.sub(r'<.*?>', '', t)
                    #     # if not t:
                    #     #     t = ' '
                    #     # tokens_lists[ind][k]=t
                    #     print()
                    #     tokens_lists[ind][k]=" "
                    #     embed()
                    if "[" in tk or "]" in tk:
                        tokens_lists[ind][k] = re.sub(r"\[+", '[', tokens_lists[ind][k])
                        tokens_lists[ind][k] = re.sub(r"\]+", ']', tokens_lists[ind][k])
                    tokens_lists[ind][k] = tokens_lists[ind][k].replace("\"", "")
                    # tokens_lists[ind][k] = tokens_lists[ind][k].replace('\\',"").replace('\"',"").replace('\/',"")


            tokens_lists_tmp = tokens_lists[j:(j+bsz)] if (j+bsz<len(tokens_lists)) else tokens_lists[j:len(tokens_lists)]

            # for tk in tokens_lists_tmp:
            #     if len(tk)>60:
            #         print("\n", tk)
            # embed()
            # print('tokens_lists_tmp',tokens_lists_tmp)

            # for cur_tmp_sent in tokens_lists_tmp:
            #     print("sent", cur_tmp_sent)
            #
            #     try:
            #         amr_list_tmpp = amr_parse([cur_tmp_sent], batch_size=batch_size, roberta_batch_size=roberta_batch_size, parser=parser)
            #     except Exception as e:
            #         print(e)
            #         embed()

            amr_list_tmp=amr_parse(tokens_lists_tmp, batch_size=batch_size, roberta_batch_size=roberta_batch_size, parser=parser)

            amr_list.extend(amr_list_tmp)

            amr_graphs_tmp, align_tmp, exist_tmp, _ = processing_amr(amr_list_tmp, tokens_lists_tmp)
            amr_graphs.extend(amr_graphs_tmp)
            align.extend(align_tmp)
            exist.extend(exist_tmp)

            if "_train" in output_path:
                dump_file([amr_graphs, align, exist, amr_list], os.path.join(get_directory(output_path), "amr_train.pkl"))
                dump_file([amr_graphs, align, exist, amr_list], os.path.join(get_directory(output_path), "amr_train.json"))
            elif "_dev" in output_path:
                dump_file([amr_graphs, align, exist, amr_list], os.path.join(get_directory(output_path), "amr_dev.pkl"))
                dump_file([amr_graphs, align, exist, amr_list], os.path.join(get_directory(output_path), "amr_dev.json"))
            elif "_test" in output_path:
                dump_file([amr_graphs, align, exist, amr_list], os.path.join(get_directory(output_path), "amr_test.pkl"))
                dump_file([amr_graphs, align, exist, amr_list], os.path.join(get_directory(output_path), "amr_test.json"))
            print("j finished",j)

    """====tree for record paths for scripts===="""

    tree=OrderedDict() # restructure scattered steps/titles/subgoals
    cur_title_pos=0
    for i, event in enumerate(samples):
        step_type=event['step_type']
        if step_type=="goal":
            cur_title_pos=i
            continue

        doc_id=event['doc_id']
        path_id=event['path_id']
        if doc_id not in tree:
            tree[doc_id]=OrderedDict()
        if path_id not in tree[doc_id]:
            tree[doc_id][path_id]=[cur_title_pos]
        tree[doc_id][path_id].append(i)
    # print("tree")
    # print(tree[0])
    # print(tree[1])

    # small test
    keys=list(tree.keys())
    for i, key in enumerate(keys):
        if i==len(keys)-1:break
        if keys[i+1]<=key:
            print("i",i)
            print("\n\nkeys[i+1]", keys[i+1])
            print("key", key)
            # breakpoint()



    """====create data samples===="""
    tmp=[]
    full_seqs=[]  # record all paths (full paths)
    # print("tree", tree)
    for doc_id in tree:
        # print("doc_id", doc_id)
        for path_id in tree[doc_id]:
            # print("path_id", path_id)
            cur_path=tree[doc_id][path_id]
            # print("cur_path", cur_path)

            cur_concept_id_to_pos_list=[]
            cur_qc=set()
            cur_doc=[]
            cur_new_doc=[]
            cur_steps=[]
            cur_step_types=[]
            cur_graphs=[]
            # last_step_qc=set()
            goal_qc=None
            categories=None

            all_steps=[samples[pos]["step"] for k, pos in enumerate(cur_path)]+["finished"]
            all_step_types=[samples[pos]["step_type"] for k, pos in enumerate(cur_path)]
            for k, pos in enumerate(cur_path):
                # print("pos", pos)
                if k==0:
                    if collect_ccpts:
                        goal_qc=set(res[pos]['qc'])
                        print("goal_qc", goal_qc)
                    categories=samples[pos]["categories"]

                if collect_ccpts:
                    cur_qc.update(res[pos]['qc'])
                    cur_concept_id_to_pos_list.append(res[pos]['concept_id_to_pos_list'])

                # new_concept_id_to_pos_list=res[pos]['concept_id_to_pos_list']
                # for ccpt_id in new_concept_id_to_pos_list:
                #     if ccpt_id not in cur_concept_id_to_pos_list:
                #         cur_concept_id_to_pos_list[ccpt_id]=new_concept_id_to_pos_list
                #     else: cur_concept_id_to_pos_list[ccpt_id].update(new_concept_id_to_pos_list)

                cur_steps.append(samples[pos]["step"])
                cur_step_types.append(samples[pos]["step_type"])
                if do_est:
                    cur_new_doc.append(samples[pos]["new_doc"])
                if do_amr:
                    cur_doc.append(tokens_lists[pos])
                if not full_seq_only and do_amr:
                    cur_graphs.append(amr_graphs[pos])

                if k == 0 and samples[pos]["doc_type"]=="methods":
                    continue # don't need title to predict subgoal for methods-type articles
                tmp.append({
                    "doc_id": doc_id,
                    'global_doc_id': samples[pos]["global_doc_id"],
                    "path_id": path_id,
                    "doc_type": samples[pos]["doc_type"],
                    "categories": categories, #samples[pos]["categories"],
                    # "url":samples[pos]["url"],
                    "cur_steps": cur_steps.copy(),
                    'step_types':cur_step_types.copy(),
                    "tgt_text": samples[cur_path[k+1]]["step"] if k<len(cur_path)-1 else "finished",
                    "tgt_steps": all_steps[(k+1):],
                    "tgt_step_types": all_step_types[(k+1):],
                    "doc_list": cur_doc.copy(),##############
                    "new_doc_list": cur_new_doc.copy(),##############

                })

                if not full_seq_only and do_amr:
                    tmp[-1].update({
                        "g_data": cur_graphs.copy(),
                    })
                if collect_ccpts:
                    tmp[-1].update({
                    "qc": sorted(cur_qc.copy()),
                    "first_qc": sorted(goal_qc.copy()),
                    "last_qc": sorted(set(res[pos]['qc'])) if k>0 else [],
                    "ac": [],
                    "concept_id_to_pos_list": cur_concept_id_to_pos_list.copy(),
                    })

                if k==len(cur_path)-1:
                    full_seqs.append(tmp[-1])
    res=tmp
    if not full_seq_only and do_amr:

        print("clean amr ROOT")

        for output_data in [res, full_seqs]:
            for item in tqdm(output_data):
                for i, g in enumerate(item['g_data']):
                    if item['doc_list'][i][-1] == "<ROOT>":
                        print("has root")
                        for s, e in g['x']:
                            if s == len(item['doc_list'][i]) - 1:
                                print('i',i)
                                print(item)
                                breakpoint()
                        # print(item)
                        item['doc_list'][i].pop()
                        # print(item['doc_list'])
                    # if g['root'] == -1 or not len(g['x']):
                    #     breakpoint()

        print("len(res3)", len(res))
        # aggregate back to path as a sample
        print("get_directory(output_path)", get_directory(output_path))


    if "_train" in output_path or full_seq_only:
        dump_file(full_seqs, os.path.join(get_directory(output_path), "full_seqs.json"))
        print("full seq saved")

    dump_file(res, output_path)
    print(f'grounded concepts saved to {output_path}')
    print()


# noinspection PyTypeChecker
def ground_sents_direct(samples, cpnet_vocab_path, pattern_path, output_path, num_processes=1, debug=False, use_pool=True, cur_steps=None, cur_qc=None, cur_doc=None, cur_concept_id_to_pos_list=None):
    global PATTERN_PATH, CPNET_VOCAB
    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)

    # samples=load_file(statement_path)

    # Input src text is list of steps
    # sents = [" [SEP] ".join(item["src_text"]) for item in samples]
    sents = samples
    answers = ["" for _ in samples] # Dummy var, won't be used
    # print(sents[:2])
    # print(answers[:2])

    # with open(statement_path, 'r') as fin:
    #     lines = [line for line in fin]
    #
    # if debug:
    #     lines = lines[192:195]
    #     print(len(lines))
    # for line in lines:
    #     if line == "":
    #         continue
    #     j = json.loads(line)
    #     # {'answerKey': 'B',
    #     #   'id': 'b8c0a4703079cf661d7261a60a1bcbff',
    #     #   'question': {'question_concept': 'magazines',
    #     #                 'choices': [{'label': 'A', 'text': 'doctor'}, {'label': 'B', 'text': 'bookstore'}, {'label': 'C', 'text': 'market'}, {'label': 'D', 'text': 'train station'}, {'label': 'E', 'text': 'mortuary'}],
    #     #                 'stem': 'Where would you find magazines along side many other printed works?'},
    #     #   'statements': [{'label': False, 'statement': 'Doctor would you find magazines along side many other printed works.'}, {'label': True, 'statement': 'Bookstore would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Market would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Train station would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Mortuary would you find magazines along side many other printed works.'}]}
    #
    #     for statement in j["statements"]:
    #         sents.append(statement["statement"])
    #
    #     for answer in j["question"]["choices"]:
    #         ans = answer['text']
    #         # ans = " ".join(answer['text'].split("_"))
    #         try:
    #             assert all([i != "_" for i in ans])
    #         except Exception:
    #             print(ans)
    #         answers.append(ans)

    res = match_mentioned_concepts(sents, answers, num_processes, sent_only=True)
    res = prune(res, cpnet_vocab_path)


    for k, item in enumerate(res):

        cur_qc[k].update(item['qc'])

        new_concept_id_to_pos_list = item['concept_id_to_pos_list']
        for ccpt_id in new_concept_id_to_pos_list:
            if ccpt_id not in cur_concept_id_to_pos_list:
                cur_concept_id_to_pos_list[k][ccpt_id] = new_concept_id_to_pos_list
            else:
                cur_concept_id_to_pos_list[k][ccpt_id].update(new_concept_id_to_pos_list)

        cur_doc[k].append(item['doc'])
        cur_steps[k].append(samples[k]["step"])

        # tmp.append({
        #     "doc_id": doc_id,
        #     "path_id": path_id,
        #     "doc_type": samples[pos]["doc_type"],
        #
        #     "doc_list": cur_doc,
        #     "cur_steps": cur_steps,
        #     "qc": cur_qc,
        #     "concept_id_to_pos_list": cur_concept_id_to_pos_list,
        #     "tgt_text": samples[cur_path[k + 1]]["step"] if k < len(cur_path) - 1 else "finished",
        # })


    # check_path(output_path)
    # print("res", res)
    return cur_steps, cur_qc, cur_doc, cur_concept_id_to_pos_list




if __name__ == "__main__":
    create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    # ground("../data/statement/dev.statement.jsonl", "../data/cpnet/concept.txt", "../data/cpnet/matcher_patterns.json", "./ground_res.jsonl", 10, True)

    # s = "a revolving door is convenient for two direction travel, but it also serves as a security measure at a bank."
    # a = "bank"
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # ans_words = nlp(a)
    # doc = nlp(s)
    # ans_matcher = Matcher(nlp.vocab)
    # print([{'TEXT': token.text.lower()} for token in ans_words])
    # ans_matcher.add("ok", None, [{'TEXT': token.text.lower()} for token in ans_words])
    #
    # matches = ans_matcher(doc)
    # for a, b, c in matches:
    #     print(a, b, c)
