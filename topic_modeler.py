#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##MIT LICENSE

#Copyright (c) 2018 Muhammad Zuhayr Raghib

#Permission is hereby granted, free of charge, to any person
#obtaining a copy of this software and associated documentation
#files (the "Software"), to deal in the Software without
#restriction, including without limitation the rights to use,
#copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the
#Software is furnished to do so, subject to the following
#conditions:

#The above copyright notice and this permission notice shall be
#included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#OTHER DEALINGS IN THE SOFTWARE.

import json
from helper import preprocess, import_json
import numpy as np
import sys
import gensim
np.random.seed(2018)


def prediciton(input_file, 
               output_file, 
               model_file = './models/model', 
               dict_file = './dictionaries/dictionary'):
    '''
    Generates prediction written to an output file in the following format:         
        
    {"_id": "abcdef", "topics": ["topic1", "topic2", "topic3", "topic4", "topic5”]}    
    
    '''
    #load model
    lda_model = gensim.models.LdaMulticore.load(model_file)
    
    #load dictionary
    dictionary = gensim.corpora.Dictionary.load(dict_file)    
    
    
    dict_ =  import_json(input_file)
    _id = dict_['_id']
    text = dict_['text']
    
    bow_vector = dictionary.doc2bow(preprocess(text))
    
    # Save top 5 topics
    output_data = {}
    topics = []
    topic_length = 5
    count = 0
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        if(count == 5):
            break
        topics.append(lda_model.print_topic(index, topic_length))
        count+=1

    output_data["_id"] = _id
    output_data["topics"] = topics

    # write to output file
    with open(output_file, 'w') as data_file:
        json.dump(output_data, data_file)
    
        
    
if __name__ == "__main__": 
    input_file = sys.argv[1]
    output_file = sys.argv[2]     
    
    prediciton(input_file, output_file)    
    

   


