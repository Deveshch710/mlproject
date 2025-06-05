#we do dtaa transformation and validation here for label encoding all thimgs here
#We code is used to transform the data and validate it


import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging 
import os

from src.utlis import save_object

#class which provide the input things for the data transformation. We are making pipelines for the ml model so that it automated and we dont have to make it again and again
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


#class which is used to do the actuall data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    #this funtion is used to converting categorical columns to numerical columns and scaling the numerical columns 
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            #numerical pipeline which do these 2 task on the training dataset
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            #categorical pipeline which do these 2 task on the training dataset
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            #column transformer which combine both the pipelines
            preproscessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preproscessor

        except Exception as e:
            raise CustomException(e, sys)
            
        
    #funtion which has data tranformation techniques
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")


            logging.info("Obtaining preprocessor object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr= np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info("Saved preprocessing object")

            save_object (
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
                

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)