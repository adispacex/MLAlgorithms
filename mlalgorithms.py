#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score


# Creation of sample data for prediction ML Algorithms
#The sample data includes 4 features 'Transaction Volume', 'Market Sentiment','Mining Difficulty','Active Address' for a Cryptocurrency assuming 'Blockchain'
#One Target variable,'Price Change' is continuous and will be predicted using Linear Regression.
data_prediction = {
    'Transaction Volume': [3000, 3200, 2800],
    'Market Sentiment': [0.75, 0.80, 0.65],
    'Mining Difficulty': [12.5, 12.7, 12.3],
    'Active Addresses': [80000, 82000, 78000],
    'Price Change': [2.5, -1.2, 0.8]
}
#The above given dataset is too small for the algorithms to predict accurately
#This will cause over-fitting or under-fitting of the model
#So I have randomly generated 1000 more rows which will be appended in the dataset given above
#Depending on the type of data for each feature or target, I generated the random data accordingly by giving the range suitable for it too
for _ in range(5000):  # Increase the number of samples as needed
    
    data_prediction['Transaction Volume'].append(np.random.randint(1000, 5000))
    data_prediction['Market Sentiment'].append(np.random.uniform(0, 1))
    data_prediction['Mining Difficulty'].append(np.random.uniform(10, 15))
    data_prediction['Active Addresses'].append(np.random.randint(70000, 90000))
    data_prediction['Price Change'].append(np.random.uniform(-5, 5))

   


#Create the DataFrame because it gets created in a list format
#Dataframe structure is much more convenient while doing ML
df_prediction = pd.DataFrame(data_prediction)

#Split the data into features (X) and target (y), axis=1 implies 'column'

X_prediction = df_prediction.drop('Price Change', axis=1)  
y_prediction = df_prediction['Price Change']

# Split the data into training and test sets, traning size = 80%, test size = 20%
#If you don't specify a random_state value when splitting your dataset, the split will be different every time you run the split operation. 
X_train_prediction, X_test_prediction, y_train_prediction, y_test_prediction = train_test_split(X_prediction, y_prediction, test_size=0.2, random_state=42)

#Creation of sample data for Classification ML algorithms
#The sample data includes 4 features 'Transaction Volume', 'Market Sentiment','Mining Difficulty','Active Address' for a Cryptocurrency assuming 'Blockchain'
#The target variable 'Price Change' here is discrete data type where it consists of 2 classes - '0 and 1', where 0 - no price change and 1 - there is a price change
data_classification = {
    
    'Transaction Volume': [3000, 3200, 2800],
    'Market Sentiment': [0.75, 0.80, 0.65],
    'Mining Difficulty': [12.5, 12.7, 12.3],
    'Active Addresses': [80000, 82000, 78000],
    'Price Change': [1, 0, 1]  # 1 for positive, 0 for negative
}

#The above given dataset is too small for the algorithms to predict accurately
#This will cause over-fitting or under-fitting of the model
#So I have randomly generated 1000 more rows which will be appended in the dataset given above
#Depending on the type of data for each feature or target, I generated the random data accordingly by giving the range suitable for it too
for _ in range(1000):  # Adjust the number of samples as needed
    
    data_classification['Transaction Volume'].append(np.random.randint(1000, 5000))
    data_classification['Market Sentiment'].append(np.random.uniform(0, 1))
    data_classification['Mining Difficulty'].append(np.random.uniform(10, 15))
    data_classification['Active Addresses'].append(np.random.randint(70000, 90000))
    data_classification['Price Change'].append(np.random.choice([0, 1]))


#Create the DataFrame because it gets created in a list format
#Dataframe structure is much more convenient while doing ML
df_classification = pd.DataFrame(data_classification)

# Split the data into features (X) and target (y), axis = 1 implies 'column'
X_classification = df_classification.drop('Price Change', axis=1)
y_classification = df_classification['Price Change']

# Split the data into training and test sets, traning size = 80%, test size = 20%
#If you don't specify a random_state value when splitting your dataset, the split will be different every time you run the split operation.
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# data dictionary is created to hold different layers. And the first layer of the dictionary contains two keys, 'Supervised Learning' and 'Unsupervised Learning'
# The second layer after that contained two types of algorithms for supervised learning,'Classification' and 'Regression'. One type of Algorithms is given for Unsupervised Learning, 'Clustering'
# And for each type of algorithm such as classification or regression or clustering, the relevant machine learning algorithms are given inside the dictionary
# and for each machine learning algorithm that is given such as Decision tree, Svm etc., the real-world applications of the same is given in a list format
machine_learning_algorithms = {
    'Supervised Algorithms': {
        'Classification': {
            'Logistic Regression': 
            ['Traffic Accident Severity Prediction', 'Employee Attrition Prediction', 'Online Ad Click Prediction', 'Disease Risk Assessment'],
            'Support Vector Machine': 
            ['Text Classification', 'Remote Sensing', 'Stock Market Prediction', 'Gesture Recognition'],
            'Decision Tree':
             ['Retail Inventory Management', 'Airline Delay Prediction', 'Loan Default Prediction', 'Criminal Justice', 'Customer Churn Prediction'],
            'Naive Bayes': 
            ['Document Classification', 'Weather Prediction', 'Language Identification', 'Market Basket Analysis'],
            'K-Means Algorithm': 
            ['Geographical Data Clustering', 'Genetic Clustering', 'Sensor Data Analysis', 'Social Network Analysis'],
        },
        'Regression': {
            'Random Forest Regression': 
            ['Energy Consumption Forecasting', 'Retail Sales Prediction', 'Real Estate Valuation', 'Medical Diagnosis', 'Air Quality Prediction'],
            'Neural Network Regression': 
            ['Game AI and Simulation', 'Financial Risk Management', 'Autonomous Vehicles', 'Stock Price Prediction', 'Demand Forecasting'],
            'Bayesian Regression': 
            ['Estimating Asset Returns', 'Pharmacokinetics and Drug Development', 'Remote Sensing and Earth Sciences', 'Real Estate Valuation', 'Supply Chain Management'],
            'Linear Regression': 
            ['Crop Yield Prediction', 'Risk Assessment', 'Air Quality Prediction', 'Marketing ROI Analysis', 'Salary Prediction'],
        }
    },
    'Unsupervised Algorithms': {
        'Clustering': {
            'Fuzzy C-Means clustering': 
            ['Image Segmentation', 'Biomedical Data Analysis', 'Pattern Recognition', 'Customer Segmentation for Marketing', 'Environmental Monitoring'],
            'Agglomerative Clustering': 
            ['Taxonomy and Phylogenetic Analysis', 'Neuroscience and Brain Mapping', 'Ecological Community Classification', 'Time Series Analysis', 'Web Page Hierarchy'],
            'Spectral Clustering':
             ['Semantic Segmentation in Computer Vision', 'Wireless Sensor Networks', 'Shape Matching', 'Graph Partitioning', 'Community Detection in Social Networks'],
            'Gaussian Mixture Models': 
            ['Speech Recognition', 'Quality Control', 'Image and Video Compression', 'Climate Modeling'],
            'BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)': 
            ['Network Traffic Analysis', 'Image Database Organization', 'E-commerce Recommendation Systems', 'Marketing ROI Analysis', 'Genetic Data Clustering'],
        }
    }
}

# Streamlit app
def main():
    # Title of the topic is mentioned
    st.title("DELVE INTO MACHINE LEARNING")

    # Step 1: Select the category of algorithms from the available keys in the machine_learning_algorithms dictionary
    selected_category = st.selectbox("**Select Category:**",list(machine_learning_algorithms.keys()))

    # Check if the selected category exists in the machine_learning_algorithms dictionary
    if selected_category in machine_learning_algorithms:
        # Step 2: Select a specific type of algorithm from the selected category type's keys
        selected_type = st.selectbox("**Select type of Algorithm:**", list(machine_learning_algorithms[selected_category].keys()))

        # Check if the selected type of algorithm exists within the selected category
        if selected_type in machine_learning_algorithms[selected_category]:
            # Step 3: Get the algorithms associated with the selected type of algorithm
            selected_algorithm = st.selectbox("**Select Algorithm:**",list(machine_learning_algorithms[selected_category][selected_type].keys()))

            # Step 4: List the real-world applications of each algorithm for the given key 
            st.write("**Real World Applications for the selected Algorithm:**")
            count = 0
            # Loop through the list in the nested dictionary to list down all the real-world examples
            # selected_use_cases is a variable to hold the use cases or real-world applications
            selected_use_cases = machine_learning_algorithms[selected_category][selected_type][selected_algorithm]
            for index,item in enumerate(machine_learning_algorithms[selected_category][selected_type][selected_algorithm],start=1):
                st.write(f"{index}. {item}")
                count += 1
            # Count is a variable which was assigned the value 0. After looping through the real-world example associated with each algorithm, count gets added by 1 for every real-world application.
            # This arithmetic operation is used to output the total number of real-world use cases in the given key
            st.write(f"**Total number of applications:**",count)

        # More math operations are performed here. The user is given the option to choose either to sort or reverse the order of the real-world applications or use cases.
        # Then the value of the selected_use_cases is copied into another variable, modified_use_cases
        # The input from the user whether they chose sort or reverse is handled by an if operation. If the user chose sort, it sorts modified_use_cases and loops through modified_use_cases.sort() and writes it
        # If the input is reverse, it reverses modified_use_cases and loops through modified_use_cases.reverse() and writes it
            if st.checkbox("**Perform Math Operations**"):
                operation = st.radio("Select operation:", ["Sort", "Reverse"])
                modified_use_cases = selected_use_cases.copy()

                if operation == "Sort":
                    modified_use_cases.sort()
                    st.write("**Use cases sorted:**")
                    for index,use_case in enumerate(modified_use_cases, start = 1):
                        st.write(f"{index}. {use_case}")
                elif operation == "Reverse":
                    modified_use_cases.reverse()
                    st.write("**Use cases reversed:**")
                    for index,use_case in enumerate(modified_use_cases, start = 1):
                        st.write(f"{index}. {use_case}")
                else:
                    st.write(" ") 
                
            # Slice function for arrays is incorporated.
            # User is asked to check the box if he/she wants to slice the array of real-world applications given by the selected_use_cases variable
            # User is asked for input of start_index and end_index. In start index, the minimum value assigned is 1 and the maximum value assigned is the length of the list, selected_use_cases
            # In end index, the minimum value is assigned as the start index and the maximum value is assigned as the total length of the same list, selected_use_cases
            # This way, we can avoid user errors
            # sliced_use_cases variable is created and it holds the sliced value of selected_use_cases using the start_index and end_index inputted by the user
            # Now we loop through the sliced_use_case using a for loop and write the values one by one
            if st.checkbox("**Slice Use Cases**"):
                start_index = st.number_input("Enter start index:", min_value=1, max_value=len(selected_use_cases), value=1)
                end_index = st.number_input("Enter end index:", min_value=start_index, max_value=len(selected_use_cases), value=len(selected_use_cases))

                sliced_use_cases = selected_use_cases[start_index - 1:end_index]
                st.write(f"Sliced use cases ({start_index} to {end_index}):")
                for index, use_case in enumerate(sliced_use_cases, start=start_index):
                    st.write(f"{index}. {use_case}")

                # Display the total count of sliced use cases
                st.write(f"Total number of use cases: {len(sliced_use_cases)}")
            else:
                # Display the total count of original use cases
                st.write(f"Total number of use cases: {len(selected_use_cases)}")

    else:
        st.write("Please select a valid algorithm type.")
    #User can select the checkbox if they want to run an ML algorithm on an in-built crypto-currency based dataset
    #After that, user is given the option to choose which ML algorithm to run on the built-in dataset
    #Depending on the algorithm chosen by the user, the equivalent IF condition is ran.
    #Depending on the algorithm chosen by the user, the data_predicition dataset or the data_classification dataset is utilized
    if st.checkbox("**Predict or classify target variables for a built-in Block-Chain Database**"):

        all_algorithms = ['Linear Regression', 'Decision Tree', 'Logistic Regression', 'Support Vector Machines']
        selected_algorithm_ML = st.selectbox("Select Algorithm:", list(all_algorithms))
        #For any algorithm that the user chooses, the first 10 rows of the dataframe are displayed
        #A Machine Learning model is created and fitted based on the algorithm chosen using the training dataset of X and Y, ie, features and target
        #Then, the target values of the testing dataset is predicted using the x test dataset
        #After that, the predicted values and actual values of the target variable are compared
        #Mean Squared Error(MSE) and R2 Score are used for this in case of predicition algorithm
        #Accuracy and F1 Score are used for this in case of classification algorithm
        #MSE - MSE measures how far, on average, the predicted values are from the actual values. 

        #A lower MSE indicates that the model's predictions are closer to the actual values, which is a desirable outcome. 
        #Conversely, a higher MSE suggests that the model's predictions are farther from the actual values, indicating poorer performance.

        #R2 - the proportion of the variance in the dependent variable (target) that's predictable from the independent variables (features). 

        #Accuracy - It is the proportion of total number of correctly classified values upon the total number of obervations
        # Accuracy = TP + TN / TP + TN + FP + FN

        # In imbalanced datasets, where one class significantly outnumbers the others, a high accuracy score can be misleading. 
        #In such cases, other metrics like precision, recall, F1-score can be used

        #F1 - The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, making it useful when there's a trade-off between these two metrics.
        if selected_algorithm_ML == "Linear Regression":
            st.write("**Snippet of Data**")
            st.dataframe(df_prediction.head(10))
            # Initialize the LinearRegression model
            model = LinearRegression()

            # Train the model on the training data
            model.fit(X_train_prediction, y_train_prediction)

            # Make predictions on the test data
            y_pred = model.predict(X_test_prediction)

            # Calculate Mean Squared Error
            mse = mean_squared_error(y_test_prediction, y_pred)
            st.write(f"Mean Squared Error:", mse.round(4))

            # Calculate R2 Score
            r2 = r2_score(y_test_prediction, y_pred)
            st.write(f"R2 Score:", r2.round(4))



        elif selected_algorithm_ML == "Decision Tree":
            st.write("**Snippet of Data**")
            st.dataframe(df_classification.head(10))
            model = DecisionTreeClassifier(random_state=42)

            # Train the model on the training data
            model.fit(X_train_classification, y_train_classification)

            # Make predictions on the test data
            y_class = model.predict(X_test_classification)

            # Calculate accuracy
            accuracy = accuracy_score(y_test_classification, y_class)
            st.write(f"Accuracy:", accuracy.round(4))

            #Calculate F1 Score
            f1 = f1_score(y_test_classification,y_class)
            st.write(f"F1 Score:", f1.round(4))

        elif selected_algorithm_ML == "Logistic Regression":
            st.write("**Snippet of Data**")
            st.dataframe(df_classification.head(10))
            # Initialize the LogisticRegression model
            model = LogisticRegression(random_state=42)

            # Train the model on the training data
            model.fit(X_train_classification, y_train_classification)

            # Make predictions on the test data
            y_class = model.predict(X_test_classification)
    

            # Calculate accuracy
            accuracy = accuracy_score(y_test_classification, y_class)
            st.write(f"Accuracy:", accuracy.round(4))
            #Calculate F1 Score
            f1 = f1_score(y_test_classification,y_class)
            st.write(f"F1 Score:", f1.round(4))

        elif selected_algorithm_ML == "Support Vector Machines":
            st.write("**Snippet of Data**")
            st.dataframe(df_classification.head(10))
            # Initialize the SVM model
            model = SVC(kernel='linear', random_state=42)

            # Train the model on the training data
            model.fit(X_train_classification, y_train_classification)

            # Make predictions on the test data
            y_class = model.predict(X_test_classification)
           

            # Calculate accuracy
            accuracy = accuracy_score(y_test_classification, y_class)
            st.write(f"Accuracy:", accuracy.round(4))
            #Calculate F1 Score
            f1 = f1_score(y_test_classification,y_class)
            st.write(f"F1 Score:", f1.round(4))

if __name__ == "__main__":
    main()
