# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:02:40 2019

@author: tsun04
"""



from gensim.models import Word2Vec
 

import tensorflow as tf


model = Word2Vec.load("C:\\Users\\tsun04\\event_sequence_embedding\\src\\output\\gensim-model.cpkt")




model.wv.most_similar(positive=["login_success"])




model.wv['login_success']


model.wv.similarity('login_success', 'getmortgagedetails')



model.wv.similarity('login_success', 'deletepayee')



model.wv.similarity('deleteemtrecipient', 'deletepayee')


model.wv.most_similar(positive=["deleteemtrecipient"])



model.wv.similarity('login_success', 'getbillerlogomap')
