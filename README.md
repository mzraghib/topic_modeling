# Topic modeling
topic modeling using LDA

# Dependencies
* Python3
* nltk
* gensim

# Instructions

```
python3 topic_modeler.py <input_file> <output_file>
```

This will output the results in the following format to the output json file location:
{"_id": "abcdef", "topics": ["topic1", "topic2", "topic3", "topic4", "topic5”]}



## Notes:
* Make sure the input and output files exist before running the program
* The dictionary and lda model are already generated and saved in the dictionaries and models directories respectfully. To generate them again, run:
```
python3 train.py
```
* For smaller texts, the model may return < 5 topics

