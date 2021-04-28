# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:36:37 2019

@author: tsun04
"""

import math
import re
import csv
from itertools import zip_longest
from datetime import datetime
from collections import Counter
import collections

def tokenize(input_file, encoding):
    mainlst=[]
    lst =[]
    with open(input_file, 'r', encoding=encoding) as f:
        for sent in f:
            sent = sent.lower()
            # Commented for personal reasons : sent = re.sub("[A-z0-9\'\"`\|\/\+\#\,\)\(\?\!\B\-\:\=\;\.\Â«\Â»\--\@]", '', sent)
            sent = re.findall('\w+', sent)
            for word in sent:
                lst.append(word)
            mainlst.append(lst)    
    return mainlst

filenm="C:\\Users\\tsun04\\event_sequence_embedding\\jrn_mobile_flatten_samp.csv"


#tokenize(filenm,'utf-8')


def ngrams_split(lst, n):
    
    return [' '.join(lst[i:i+n]) for i in range(len(lst)-n)]


sent = tokenize(filenm,'utf-8')



def n_gram_count(sent, n_filter, n):
    
    #Cdict = collections.defaultdict() 
    n_lines = len(sent)
    Cdict=Counter()
    for i in range(n_lines):
        
        ngram_select=ngrams_split(sent[i], n)
        Cdict = Cdict+Counter(ngram_select)
        
        print("ngram_count, processed"+str(i)+" lines!!")
    return Cdict    
        
            
counterdict=n_gram_count(sent,0,2) 

len(counterdict) 

def Write2File(counterdict):
    
    gram_file = open('bigram.data', 'w')
    
    for grm,cnt in counterdict.items():
        #print(grm)
        writout=str(grm)+","+str(cnt)
        #print(writout)
        gram_file.writelines(writout) 
        gram_file.writelines("\n") 
        print("Write2File, processed"+str(i)+" lines!!")
    gram_file.close()      

Write2File(counterdict)



if __name__ == "__main__":
    start_time = datetime.now()
    filenm="C:\\Users\\tsun04\\event_sequence_embedding\\jrn_mobile_flatten_samp.csv"
    s = n_grams_stat(filenm,'utf-8', n_filter=0, n=4)
    
    for a, b, c, d, e in s:
        print(a, b, c, d, e)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))