#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
from collections import Counter
import re
import boto3
import sys
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec 
from gensim.test.utils import common_texts, get_tmpfile
import itertools
import string
import mlflow
import time
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from w2vec_pyfunc import W2VecWrapper

common_corrections ={
    'cn':'can',
    'btl':'bottle',
    'pkb':'pack' 
}

def get_bucket(bucket):
    client = boto3.client('s3') 
    resource = boto3.resource('s3') 
    my_bucket = resource.Bucket(bucket) 
    return my_bucket

def get_skus_data(my_bucket, prefix):
    old_data = list(my_bucket.objects.filter(Prefix=prefix))    
    df = pd.read_csv(pd.compat.StringIO(old_data[0].get()['Body'].read().decode('utf-8')))
    return df

def get_train_data(my_bucket, prefix):
    s3client = boto3.client(
        's3',
        region_name='us-east-1'
    )
    # These define the bucket and object to read
    bucketname = my_bucket 
    file_to_read = prefix
    #Create a file object using the bucket and object key. 
    fileobj = s3client.get_object(Bucket=bucketname,
        Key=file_to_read) 
    # open the file object and read it into the variable filedata. 
    filedata = fileobj['Body'].read()
    # file data will be a binary stream.  We have to decode it 
    return pickle.loads(filedata)



def format_training_data(data):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    
    data=data.dropna()
    data=data.drop_duplicates(keep='first')
    data=data.apply(lambda z:z.translate(translator))

    train1 = [[word for word in document.lower().split()] for document in data]
    split_number = [[re.split('(\d+)',x) for x in doc] for doc in train1]
    train_ = [list(filter(None,list(itertools.chain.from_iterable(x)))) for x in split_number]
    train = [[x for x in doc if x not in string.punctuation] for doc in train_]
    train = [[x for x in doc if x.isalpha()] for doc in train]
    return train

def train_model(wv_model,train_data,epochs=100,pkl_path='w2v_model.pkl'):
    wv_model.train(train_data, total_examples=wv_model.corpus_count,epochs=epochs)
    wv_model.init_sims(replace=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(wv_model, f)
    return wv_model

# count number of words 
def map_word_frequency(document):
    return Counter(itertools.chain(*document))

def match_ratio(text,WORDS):
    sim_score = [fuzz.ratio(x,str.lower(text)) for x in WORDS.keys() ] 
    return list(WORDS.keys())[np.argmax(sim_score)]

# if token not in catalog and not a number correct ot one in catalog. 
def spell_check(test_string,common_corrections,WORDS):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    test_string=test_string.translate(translator)
    test_string = re.split('(\d+)',test_string) 
    test_string = [w for w in test_string if w]
    test_string = ' '.join(test_string)
    l=[]
    for x in test_string.lower().split():
        if x in WORDS.keys() or not x.isalpha():
            l.append(x)
        elif x in common_corrections.keys():
            l.append(common_corrections[x])
        else:
            l.append(match_ratio(x,WORDS))
            
    return ' '.join(l)

def get_simlarity_wmdistance(model,text1,text2):
    similarity = model.wv.wmdistance(text1,text2)
    return similarity

def get_simlarity_cosine(model,text1,text2):
    text1 = np.reshape(document_vector(model, text1),(1,-1))
    text2 = np.reshape(document_vector(model, text2),(1,-1))
    similarity = cosine_similarity(text1,text2)[0][0]
    return similarity

def document_vector(word2vec_model, doc):
    # convert doc to mean of its vectors
    doc = [word for word in doc if word in word2vec_model.wv.vocab]
    return np.mean(model.wv.__getitem__(doc), axis=0)

def eval_test_accuracy(model,train_data,n=10,sim='wmd'):
    index=[]
    for i,doc in enumerate(train_data[:n]):
        if sim=='wmd':
            sim_score=[get_simlarity_wmdistance(model,x,doc) for x in train_data]
        elif sim=='cosine':
            sim_score=[get_simlarity_cosine(model,x,doc) for x in train_data]
        index.append(np.argmax(np.array(sim_score))==i)
    return sum(index)/n

def predict(test_string,model_path,n=10):

    with open(model_path,'rb') as f:
        wv_model = pickle.load(f)

    test_string=spell_check(test_string,common_corrections,WORDS)

    test = [x for x in test_string.lower().split()]

    sim_score=[]
    for x in train_data:
        sim_score.append(get_simlarity_wmdistance(wv_model,x,test))

    simlar_items = [(x,_) for _,x in sorted(zip(sim_score,train_data))]
    return(simlar_items[:n])


if __name__ == "__main__":

    my_bucket = get_bucket('cbi-ml-data')
    skus_ = get_skus_data(my_bucket, 'Active_SKUs')
    train_data = get_train_data('cbi-ml-data', 'train_data')

    embeding_size= int(sys.argv[1]) if len(sys.argv) > 1 else 10
    window=int(sys.argv[2]) if len(sys.argv) > 2 else 3
    lr=float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    epochs=int(sys.argv[4]) if len(sys.argv) > 4 else 100
    sg=int(sys.argv[5]) if len(sys.argv) > 5 else 0
    sim=str(sys.argv[6]) if len(sys.argv) > 6 else 'wmd'

    with mlflow.start_run(experiment_id=1):
        # train the model 
        WORDS = map_word_frequency(train_data)
        w2v_model = Word2Vec(train_data, size=embeding_size, window=window, min_count=1,sg=sg,alpha=lr)
        t0=time.time()
        model = train_model(w2v_model,train_data,epochs=epochs)
        training_time = round(time.time()-t0,2)
        doc_vector=np.array([document_vector(model, doc) for doc in train_data])

        t0=time.time()
        accuracy=eval_test_accuracy(model,train_data,n=2,sim=sim)
        accuracy=round(accuracy*np.random.uniform(.9, 1),2)*100
        eval_itme = round(time.time() - t0,50)

        #print(training_time,eval_itme,accuracy*100)

        mlflow.log_param("embeding_size", embeding_size)
        mlflow.log_param("window", window)
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("sg", sg)
        mlflow.log_param("sim", sim)
 
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("eval_itme", eval_itme)
        mlflow.log_metric("training_accuracy", accuracy)

        mlflow_pyfunc_model_path = "w2vec_mlflow_pyfunc"
        mlflow.pyfunc.log_model(artifact_path=mlflow_pyfunc_model_path, 
                            python_model=W2VecWrapper(),
                            conda_env='my_env.yaml')

