import mlflow.pyfunc
import pandas as pd 
import numpy as np
from collections import Counter
import boto3
import sys
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec 
from gensim.test.utils import common_texts, get_tmpfile
import itertools
import string
import mlflow
import re
import pickle

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

class W2VecWrapper(mlflow.pyfunc.PythonModel):
    
    def match_ratio(self,text):
        sim_score = [fuzz.ratio(x,str.lower(text)) for x in self.wv_model.wv.vocab.keys() ] 
        return list(self.wv_model.wv.vocab.keys())[np.argmax(sim_score)]

    # if token not in catalog and not a number correct ot one in catalog. 
    def spell_check(self,test_string):
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        test_string = test_string['predict'].values[0]
        test_string = test_string.translate(translator)
        test_string = re.split('(\d+)',test_string) 
        test_string = [w for w in test_string if w]
        test_string = ' '.join(test_string)
        
        l=[]
        
        for x in test_string.lower().split():
            if x in self.wv_model.wv.vocab.keys() or not x.isalpha():
                l.append(x)
            elif x in self.common_corrections.keys():
                l.append(self.common_corrections[x])
            else:
                l.append(self.match_ratio(x))

        return ' '.join(l)
    
    def get_simlarity_wmdistance(self,model,text1,text2):
        similarity = model.wv.wmdistance(text1,text2)
        return similarity

    def load_context(self,context):
        with open('w2v_model.pkl', "rb") as f:
            model = pickle.load(f)
        self.wv_model = model
        self.common_corrections={'cn':'can',
                                'btl':'bottle',
                                'pkb':'pack'}                  
        self.train_data = get_train_data('cbi-ml-data', 'train_data')

    def predict(self,context,test_string,n=10):
        test_string=self.spell_check(test_string)
        test = [x for x in test_string.lower().split()]
        sim_score=[]
        for x in self.train_data:
            sim_score.append(self.get_simlarity_wmdistance(self.wv_model,x,test))
        simlar_items = [(x,_) for _,x in sorted(zip(sim_score,self.train_data))]
        return(simlar_items[:n])