# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:02:49 2019

@author: tsun04

revised from https://github.com/jbhoosreddy/ngram/blob/master/ngram.py

"""

from collections import Counter
import math as calc
import gc

class nGram():
    """A program which creates n-Gram (1-5) Maximum Likelihood Probabilistic Language Model with Laplace Add-1 smoothing
    and stores it in hash-able dictionary form.
    n: number of bigrams (supports up to 5)
    corpus_file: relative path to the corpus file.
    cache: saves computed values if True
Usage:
>>> ng = nGram(n=5, corpus_file=None, cache=False)
>>> print(ng.sentence_probability(sentence='hold your horses', n=2, form='log'))
>>> -18.655540764
"""
    def __init__(self, n=1, corpus_file=None, cache=False):
        """Constructor method which loads the corpus from file and creates ngrams based on imput parameters."""
        self.words = []
        self.load_corpus(corpus_file)
        self.unigram = self.bigram = self.trigram = self.quadrigram = self.pentigram = self.tengram = None
        
        '''
        self.create_unigram(cache)
        
        if n >= 2:
            self.create_bigram(cache)
        if n >= 3:
            self.create_trigram(cache)
           
        if n >= 4:
            self.create_quadrigram(cache)
            ''' 
        if n >= 5:
            self.create_pentigram(cache)
           
        if n == 10:
            self.create_tengram(cache)
             
            
        #self.print_mostcommon()    
        return
    
        
    

    def load_corpus(self, file_name):
        """Method to load external file which contains raw corpus."""
        print("Loading Corpus from data file")
        if file_name is None:
            file_name = "corpus.data"
        corpus_file = open(file_name, 'r')
        corpus = corpus_file.read()
        corpus_file.close()
        print("Processing Corpus")
        
        self.words = corpus.split(' ')
        self.corpus=corpus
    
    def create_unigram(self, cache):
        """Method to create Unigram Model for words loaded from corpus."""
        print("Creating Unigram Model")
        unigram_file = None
        if cache:
            unigram_file = open('unigram.data', 'w')
        print("Calculating Count for Unigram Model")
        
        uniwords=[]
        
        for word in self.words:
            if word.find("\n")<0:
                uniwords.append(word)
        
        print("Use counter method for cleaned words")        
        unigram = Counter(uniwords)
        
        print("length of the unigram counter: ") 
        print(len(set(unigram.keys())))
        if cache:
            unigram_file.write(str(unigram))
            #unigram_file.write("\n")
            unigram_file.close()
        self.unigram = unigram
        
        #for elem in unigram.most_common(10):
        #    print(elem)

    def create_bigram(self, cache):
        """Method to create Bigram Model for words loaded from corpus."""
        print("Creating Bigram Model")
        words = self.words
        biwords = []
        for index, item in enumerate(words):
            if index == len(words)-1:
                break
            biwords.append(item+' '+words[index+1])
        print("Calculating Count for Bigram Model")
        bigram_file = None
        if cache:
            bigram_file = open('bigram.data', 'w')
            
        biwords_rev=[]
        
        for word in biwords:
            if word.find("\n")<0:
                biwords_rev.append(word)
        
        print("Use counter method for cleaned words")     
        bigram = Counter(biwords_rev)
        
        print("length of the bigram counter: ") 
        print(len(set(bigram.keys())))
        
        if cache:
            bigram_file.write(str(bigram))
            bigram_file.close()
        self.bigram = bigram
        
        #print(bigram.most_common(10))

    def create_trigram(self, cache):
        """Method to create Trigram Model for words loaded from corpus."""
        print("Creating Trigram Model")
        words = self.words
        triwords = []
        for index, item in enumerate(words):
            if index == len(words)-2:
                break
            triwords.append(item+' '+words[index+1]+' '+words[index+2])
        print("Calculating Count for Trigram Model")
        if cache:
            trigram_file = open('trigram.data', 'w')
            
        triwords_rev=[]
        
        for word in triwords:
            if word.find("\n")<0:
                triwords_rev.append(word)
        
        print("Use counter method for cleaned words")     
        trigram = Counter(triwords_rev)
        
        print("length of the trigram counter: ") 
        print(len(set(trigram.keys())))

        if cache:
            trigram_file.write(str(trigram))
            trigram_file.close()
        self.trigram = trigram
        
        #print(trigram.most_common(10))

    def create_quadrigram(self, cache):
        """Method to create Quadrigram Model for words loaded from corpus."""
        print("Creating Quadrigram Model")
        words = self.words
        quadriwords = []
        for index, item in enumerate(words):
            if index == len(words)-3:
                break
            quadriwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3])
        print("Calculating Count for Quadrigram Model")
        if cache:
            quadrigram_file = open('fourgram.data', 'w')
        
        quadriwords_rev=[]
        
        for word in quadriwords:
            if word.find("\n")<0:
                quadriwords_rev.append(word)
        
        print("Use counter method for cleaned words")  
        
        quadrigram = Counter(quadriwords_rev)
        
        print("length of the quadrigram counter: ") 
        print(len(set(quadrigram.keys())))
        
        if cache:
            quadrigram_file.write(str(quadrigram))
            quadrigram_file.close()
        self.quadrigram = quadrigram
        
        #print(quadrigram.most_common(10))

    def create_pentigram(self, cache):
        """Method to create Pentigram Model for words loaded from corpus."""
        print("Creating pentigram Model")
        words = self.words
        pentiwords = []
        for index, item in enumerate(words):
            if index == len(words)-4:
                break
            pentiwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3]+' '+words[index+4])
            
        print("Calculating Count for pentigram Model")
        if cache:
            pentigram_file = open('pentagram.data', 'w')
        
        
        pentiwords_rev=[]
        
        for word in pentiwords:
            if word.find("\n")<0:
                pentiwords_rev.append(word)
        
        print("Use counter method for cleaned words") 
        
        pentigram = Counter(pentiwords_rev)
        
        print("length of the pentigram counter: ") 
        print(len(set(pentigram.keys())))
        
        if cache:
            pentigram_file.write(str(pentigram))
            pentigram_file.close()
        self.pentigram = pentigram
        
        #print(pentigram.most_common(10))
        
        
    def create_tengram(self, cache):
        """Method to create Pentigram Model for words loaded from corpus."""
        print("Creating pentigram Model")
        words = self.words
        tenwords = []
        for index, item in enumerate(words):
            if index == len(words)-9:
                break
            tenwords.append(item+' '+words[index+1]+' '+words[index+2]+' '+words[index+3]+' '+words[index+4]+' '
                            +words[index+5]+' '+words[index+6]+' '+words[index+7]+' '+words[index+8]+' '+words[index+9])
            
            
        print("Calculating Count for tengram Model")
        if cache:
            tengram_file = open('tengram.data', 'w')
        
        
        tenwords_rev=[]
        
        for word in tenwords:
            if word.find("\n")<0:
                tenwords_rev.append(word)
        
        print("Use counter method for cleaned words") 
        
        tengram = Counter(tenwords_rev)
        
        if cache:
            tengram_file.write(str(tengram))
            tengram_file.close()
        self.tengram = tengram


    def print_mostcommon(self):
        #for grm in ['unigram','bigram','trigram','quadrigram','pentigram']:
        print('print most common')

        for com in self.grm.most_common(10):
                print(com)


#dir(fgram)
#help(nGram)

if __name__ == '__main__':
    
    gc.collect()
    filenm="C:\\Users\\tsun04\\event_sequence_embedding\\JRN_UNIVERSE_201909WK1_FLATTEN.csv"
    fgram=nGram(n=5, corpus_file=filenm, cache=False)
    
    ''' 
    print('print unigram most common')
    for elem in fgram.unigram.most_common(10):
            print(elem)
        
    print('print bigram most common')
    for elem in fgram.bigram.most_common(10):
            print(elem)        
     
    
    print('print trigram most common')
    for elem in fgram.trigram.most_common(10):
            print(elem) 
    
   
    print('print quadrigram most common')
    for elem in fgram.quadrigram.most_common(10):
            print(elem)        
    
    
    print('print pentigram most common')
    for elem in fgram.pentigram.most_common(10):
            print(elem) 
            
    print('print tengram most common')
    for elem in fgram.tengram.most_common(10):
            print(elem)   
           
    '''
    

#members = [attr for attr in dir(fgram) if not callable(getattr(fgram, attr)) 
#and not attr.startswith("__")]    
   
