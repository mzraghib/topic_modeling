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



from helper import import_multiple_json, preprocess
import numpy as np
import nltk


file_name = './data/topic_modeling_data.json'
list_dict =  import_multiple_json(file_name)


#Data preprocessing
import gensim

np.random.seed(2018)
nltk.download('wordnet')

    
    
def train():
    '''
    Generate the trained LDA model and a dictionary object
    
    '''

    # Preprocess all text samples
    processed_docs = [preprocess(i['text']) for i in list_dict]
    
    # Bag of words on the dataset
    dictionary = gensim.corpora.Dictionary(processed_docs)
    
    
    # filter extremes
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    
    
    #For each document we create a dictionary reporting how many
    #words and how many times those words appear. Save this to ‘bow_corpus’, then check our selected document earlier.
    
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    
    bow_doc_0 = bow_corpus[0]
    
    for i in range(len(bow_doc_0)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_0[i][0], 
                                                         dictionary[bow_doc_0[i][0]], 
                                                         bow_doc_0[i][1]))
    
    
    # Running LDA using Bag of Words
    lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                           num_topics=10,
                                           id2word=dictionary, 
                                           passes=2,
                                           workers=2)
    
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    
    return dictionary, lda_model  
        
        
    
if __name__ == "__main__":
    dictionary, lda_model = train()   
    
    # Save model to disk.
    temp_file = './models/model'
    lda_model.save(temp_file)

    # Save dictionary to disk.
    temp_file = './dictionaries/dictionary'
    dictionary.save(temp_file)
