### Overview
This repository provides a comprehensive implementation for predicting theatre procedure times in a healthcare setting. It leverages data analytics and machine learning techniques to build predictive models aimed at enhancing operational efficiency and resource planning in hospitals. The repository includes self-built modules for data processing, encoding, and predictive modeling.

Details on Problem Setting and Methodologies can be found in IEEE paper: https://ieeexplore.ieee.org/document/10803395

### Features

  #### Data Preprocessing and Encoding:
  
  Includes a custom-built module (data_processing_nd_encoding) for handling data preprocessing, feature engineering, and encoding tasks.

  This section of the code prepares the dataset for modeling purposes using two key functions from the custom module data_processing_nd_encoding_related_functions:

  ##### preprocess_dataset: 
  Handles the preprocessing of raw data, including filtering, cleaning, encoding, and handling outliers.
  ###### prepare_IO_dataset: 
  Converts the processed dataset into a format suitable for machine learning models by applying one-hot encoding and preparing input-output features.
  
  ##### Key Steps:
  Specify Dataset and Target Variables:
  
  **data_file:** Specifies the input dataset (e.g., Theatre_data_prepared_TO_2018_2024.csv).
  **target_data_types:** Defines the target variable(s) for prediction (e.g., 'H4 Minutes').
  
  ##### Define Input Features:
  
  **Essential Features:** Specified in model_input_features_ess, these are mandatory input features such as 'Actual Procedure 1 Code 1'.
  **Additional Features:** Specified in model_input_features_addi, these include contextual or auxiliary features like hospital data, patient demographics, and procedure-related information.
  
  ##### Data Filtering and Categorization:
  
  **dict_for_data_model_categorisation:** Filters the data to a specific subset, e.g., specialty as 'Trauma and Orthopaedics'.
  **filtering_data_column: ** Allows additional filtering criteria (empty in this example).
  
  ##### Handle Infrequent Categories and Outliers:
  
  **min_count_threshold:** Ensures categories with counts below the threshold (e.g., Consultant Code < 25) are replaced with fallback values, defined in infrequent_data_replacement.
  **Z_score_for_outlier_removal:** Specifies z-score thresholds for removing outliers globally or for specific features.
  
  ##### Preprocess Dataset:
  The preprocess_dataset function is called with the configurations specified above to clean, filter, and preprocess the dataset.
  
  ##### Prepare Input-Output Dataset:
  The prepare_IO_dataset function is used to transform the preprocessed dataset into model-ready input-output datasets with one-hot encoding applied to specified categorical features.

  #### Predictive Modeling:

  Utilizes a range of machine learning algorithms for theatre time prediction, with support for performance evaluation and model optimization.
  
  Contains reusable functions for predictive modeling (predictive_model_related_functions).


#### Notebook for Model Training and Testing:

  A Jupyter Notebook (Training_and_testing_multiple_models_TO.ipynb) is provided to guide users through training, testing, and evaluating multiple models.

### Prerequisites

To run the project, ensure you have the following installed:

1. Python 3.x

Required Python packages:

- pandas

- numpy

- scikit-learn

- matplotlib

- seaborn

Any additional dependencies listed in the requirements.txt file (to be included).
