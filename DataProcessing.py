
import pandas as pd
import pickle
import numpy as np


class PreProcessing:
    

    def __init__(self,train_name,test_name,train_text,test_text):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        
        self.raw_data=pd.read_csv(train_name)
        self.raw_data_test=pd.read_csv(test_name)
        self.raw_text=pd.read_csv(train_text,sep='||',header=None,skiprows=0)
        self.raw_text_test=pd.read_csv(test_text,sep='||',header=None,skiprows=0)

    def process(self):
        self.raw_text["ID"],self.raw_text["Text"] = self.raw_text[0].str.split('\|\|', 1).str
        self.raw_text_test["ID"],self.raw_text_test["Text"] = self.raw_text_test[0].str.split('\|\|', 1).str
        self.raw_text=self.raw_text.drop(0,axis=1)
        self.raw_text_test=self.raw_text_test.drop(0,axis=1)
        self.raw_text=self.raw_text[self.raw_text.index!=0]
        self.raw_text_test=self.raw_text_test[self.raw_text_test.index!=0]
        self.raw_text=self.raw_text.reset_index()
        self.raw_text.drop(["index"],axis=1)
        self.raw_text_test=self.raw_text_test.reset_index()
        self.raw_text_test.drop(["index"],axis=1)

    def merging(self,train_data,train_text,test_data,test_text):
        final_train=train_data.merge(train_text,left_index=True,right_index=True)
        final_train=final_train.drop(["ID_y","index"],axis=1)
        final_train.rename(columns={'ID_x': 'ID'},inplace=True)
        final_train=final_train.reindex(columns=["ID","Gene","Variation","Text","Class"])
        final_test=test_data.merge(test_text,how="inner",left_index=True,right_index=True)
        final_test=final_test.drop(["ID_y","index"],axis=1)
        final_test.rename(columns={'ID_x': 'ID'},inplace=True)
        final_test=final_test.reindex(columns=["ID","Gene","Variation","Text"])
        return final_train,final_test


if  __name__ == '__main__':
    Data=PreProcessing("training_variants.txt","test_variants.txt","training_text.txt","test_text.txt")
    Data.process()
    final_train,final_test=Data.merging(Data.raw_data,Data.raw_text,Data.raw_data_test,Data.raw_text_test)
    with open('FinalTrain.pickle','wb') as f:
        pickle.dump(final_train,f)
    with open('FinalTest.pickle','wb') as f:
        pickle.dump(final_test,f)
