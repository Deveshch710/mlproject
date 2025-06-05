#taking data from some databasse like bring data.
#This file do ingestion like we have data in different different sources like databses, cloud, live data and to use that data in our project we will wirte the code in here so that we can have that data in here which we can use
#Basically we aree creating a data ci/cd pipeline which bring dtaa continously in automated manner

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

#Creating class so that can use the ingestion componets for different differnet data sources
@dataclass
class DataIngestionConfig:
    #defining path for stroing the raw,test,trained data in a folder and giving it path to get save with file name
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


#Class where we have doing whole data ingestion process
class DataIngestion:
    
    #funtion to assign data ingestion confi to the whole class so that it can be used
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    #Data ingestion
    def initiate_data_ingestion(self):
        logging.info("Enterd the data ingestion method config")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe df")

            #making files of the data we got
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            #now doing train test plit
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #making files of the data we got
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        


#initiating the whole data ingestion code
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)