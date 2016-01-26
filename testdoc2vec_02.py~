
# coding: utf-8

# In[1]:

description = """This file will make Doc2Vec models from the input. It will not use pretrained vectors"""
description 


# In[2]:

import gensim
import pandas as pd
import numpy as np
import sys
import os
import itertools

import sklearn
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

from scipy import stats
from bs4 import BeautifulSoup

reload(sys)
sys.setdefaultencoding("UTF-8")


# In[3]:

import multiprocessing
cores = multiprocessing.cpu_count()


# In[4]:

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text).get_text()
    words = review_text.lower().split()
    return(" ".join(words))


# In[5]:

datafile = "summary-english-truth.txt"


# In[6]:

train = pd.read_csv(datafile, header=0, delimiter="\t", quoting=1)


# In[7]:

all_text = train["text"]


# In[8]:

with open("only_tweets.txt", 'w') as out_file:
    for each_line in all_text:
        out_file.write(clean_text(each_line)+"\n")
out_file.close()


# In[9]:

doc2veclines = gensim.models.doc2vec.TaggedLineDocument('only_tweets.txt')
lines = []
for each in doc2veclines:
    lines.append(each)


# In[10]:

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict


# In[11]:

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
    
    Doc2Vec(dm=1, dm_concat=1, size=25, window=5, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=0, size=25, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=1, dm_concat=1, size=50, window=5, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=0, size=50, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=1, dm_concat=1, size=150, window=5, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=0, size=150, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=1, dm_concat=1, size=200, window=5, negative=5, hs=0, min_count=2, workers=cores),
    Doc2Vec(dm=0, size=200, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
]


# In[12]:

simple_models[0].build_vocab(lines)  # PV-DM/concat requires one special NULL word so it serves as template
simple_models[1].build_vocab(lines)
simple_models[2].build_vocab(lines)


# In[13]:

print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)
    
models_by_name = OrderedDict((str(model), model) for model in simple_models)


# In[14]:

import numpy as np
import statsmodels.api as sm
from random import sample

# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    
#def logistic_predictor_from_data(train_targets, train_regressors):
#    logit = sm.Logit(train_targets, train_regressors)
#    predictor = logit.fit(disp=0)
#    #print(predictor.summary())
#    return predictor

#def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
#    """Report error rate on test_doc sentiments, using supplied model and train_docs"""
#
#    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
#    train_regressors = sm.add_constant(train_regressors)
#    predictor = logistic_predictor_from_data(train_targets, train_regressors)
#
#    test_data = test_set
#    if infer:
#        if infer_subsample < 1.0:
#            test_data = sample(test_data, int(infer_subsample * len(test_data, b , b )))
#        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
#    else:
#        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
#    test_regressors = sm.add_constant(test_regressors)
    
#    # predict & evaluate
#    test_predictions = predictor.predict(test_regressors)
#    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
#    errors = len(test_predictions) - corrects 
#    error_rate = float(errors) / len(test_predictions)
#    return (error_rate, errors, len(test_predictions), predictor)


def error_rate_for_model(test_model, target):
    num_instances = len(target)
    dims = test_model.docvecs[0].shape[0]
    # preallocate
    train_x = np.zeros((num_instances, dims))
    
    for i in range(0, num_instances):
        train_x[i,:] = test_model.docvecs[i]
    
    scoring_function='accuracy'
    clf = svm.SVC(kernel='linear', C=100)
    scores = cross_validation.cross_val_score(clf, train_x, target, cv=10, scoring=scoring_function)
    err = (1.0 - scores.mean())
    
    return err


# In[15]:

from collections import defaultdict
best_error = defaultdict(lambda :1.0)


# In[16]:

from random import shuffle
import datetime

# In[17]:

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes


# In[ ]:




# In[18]:

print("START %s" % datetime.datetime.now())


# In[19]:

target = train["age"]
for epoch in range(passes):
    shuffle(lines)  # shuffling gets best results
    
    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(lines)
            duration = '%.1f' % elapsed()
            
        # evaluate
        eval_duration = ''
        with elapsed_timer() as eval_elapsed:
            err = error_rate_for_model(train_model, target)
        eval_duration = '%.1f' % eval_elapsed()
        
        best_indicator = ' '
        if err <= best_error[name]:
            best_error[name] = err
            best_indicator = '*' 
        print("%s%f : %i passes : %s %ss %ss" % (best_indicator, err, epoch + 1, name, duration, eval_duration))

#        if ((epoch + 1) % 5) == 0 or epoch == 0:
#            eval_duration = ''
#            with elapsed_timer() as eval_elapsed:
#                infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs, infer=True)
#            eval_duration = '%.1f' % eval_elapsed()
#            best_indicator = ' '
#            if infer_err < best_error[name + '_inferred']:
#                best_error[name + '_inferred'] = infer_err
#                best_indicator = '*'
#            print("%s%f : %i passes : %s %ss %ss" % (best_indicator, infer_err, epoch + 1, name + '_inferred', duration, eval_duration))

    print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta
    
print("END %s" % str(datetime.datetime.now()))


# In[ ]:


for onekey in best_error.keys():
    print onekey, best_error[onekey]
