import os
import sys
from src.exception import CustomeException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

RANDOM_STATE = 42

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    


class DataIngestion:
    '''
    Read data from source and save it into artifact file inside working space directory.
    '''
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")
            
            #Create file artifact if DNE
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            # Save read data from source to local working folder
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header=True)
            
            logging.info("Init train test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state = RANDOM_STATE)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data is completed")
            
            #return path of train data and test data for data transformation component to call
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomeException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.init_data_transformation(train_data,test_data)
