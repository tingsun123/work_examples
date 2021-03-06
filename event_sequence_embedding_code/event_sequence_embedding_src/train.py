# -*- coding: utf-8 -*-
"""

"""

import os
import json
import time
import logging
import argparse
import regex as re
from models import CsvConnector, TxtConnector, Word2Vec, Bigram, create_embeddings
import gensim



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default="C:\\Users\\tsun04\\event_sequence_embedding\\JRN_UNIVERSE_201909WK1_FLATTEN.csv")
    parser.add_argument("--input_type", choices=['csv', 'txt'],
                        default="csv", help="The kind of input data")
    parser.add_argument("--separator", type=str, default=',', help="csv separator.")
    parser.add_argument("--columns_to_select", type=str, default="event_sequence", help="column names comma separated.")
    parser.add_argument("--columns_joining_token", type=str, default='. ', help="join multiple columns.")
    parser.add_argument("--folder", default="./output")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min_count", type=int, default=0)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--sample", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min_alpha", type=float, default=0.0001)
    parser.add_argument("--sg", type=int, default=0)   # Training algorithm: 1 for skip-gram; otherwise CBOW.
    parser.add_argument("--hs", type=int, default=0)  #If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--cbow_mean", type=int, default=1) #If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--null_word", type=int, default=0)
    args = parser.parse_args()
    return args


def preprocessing(sentence):
    """
    Add a custom pre-processing on sentence before feeding them to word2vec
    """
    sentence = sentence.lower()
    sentence = re.sub(r'[^\P{P}\']+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = "".join(sentence).strip()
    return sentence


if __name__ == '__main__':

    
    opt = get_args()
    opt.folder = os.path.realpath(opt.folder)
    print(opt)

    '''
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    prefix = int(time.time())
    
    os.makedirs(opt.folder, exist_ok=True)
    json.dump(vars(opt), open(os.path.join(opt.folder, "opt.json"), 'w', encoding='utf-8'), indent=2)


   
   
    if opt.input_type == 'csv':
        sentence_generator = CsvConnector(filepath=opt.file,
                                          # tsun note: remove preprocessing as data is already cleaned
                                          #reprocessing=preprocessing,
                                          separator=opt.separator,
                                          columns_to_select=opt.columns_to_select.split(","),
                                          columns_joining_token=opt.columns_joining_token)
    elif opt.input_type == 'txt':
        sentence_generator = TxtConnector(filepath=opt.file, preprocessing=preprocessing)
    else:
        raise

    #generator = Bigram(sentence_generator)
    generator = sentence_generator

    w2v = Word2Vec(save_folder=opt.folder)
    w2v.fit(generator,
            size=opt.size,
            alpha=opt.alpha,
            window=opt.window,
            min_count=opt.min_count,
            max_vocab_size=opt.max_vocab_size,
            sample=opt.sample,
            seed=opt.seed,
            workers=opt.workers,
            min_alpha=opt.min_alpha,
            sg=opt.sg,
            hs=opt.hs,
            negative=opt.negative,
            cbow_mean=opt.cbow_mean,
            iter=opt.iter,
            null_word=opt.null_word)
'''
    w2vmodel = gensim.models.Word2Vec.load(".\output\gensim-model.cpkt")
    create_embeddings(gensim_model=w2vmodel, model_folder=opt.folder)


import gc
gc.collect()