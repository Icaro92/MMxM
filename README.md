# Sales Prediction based on Media Channel's Spend

---
<p align = "center">
  <img src="https://github.com/Icaro92/MMxM/assets/58118599/f246061b-ce7e-4fee-8b99-2dae49a7c580" width="500" alt="SalesDash Image">

## Objectives

The objective was to predict Sales based on a company's Marketing Investments in different media channels, such as YouTube, Facebook, and Newspapers.
This is a straightforward approach to a simplified dataset and could be easily perceived as a Linear regression exercise.
## Data Source

For this project, I used a Kaggle dataset found in the link below.
* [Advertising Dataset](https://www.kaggle.com/datasets/muyiwawilliams/advertising-dataset/data)

## Project Overview

  **1 - Descriptive Analysis**
  
  For this particular dataset, descriptive analysis was very straightforward. 
  * The dataset contains 171 lines and 4 columns
  * None of the columns had missing or null values
  * It was not necessary to alter the Dtypes of any column

  **2 - Exploratory Analysis**
  * Statistical Metrics for this dataset proved to be closely scaled and logical:
  <p align = "left">
  <img src="https://github.com/Icaro92/MMxM/assets/58118599/24d15365-9e68-4a37-8b72-2cd195360d47" width="500" alt="SalesDash Image">

  * Verifying statistical linearity of the multiple variables through a Pairplot.
  * Facebook and YouTube have correlations closer to linear with the sales variable and are likely to have more influence over Sales than newspapers.
    
  <p align = "left">
  <img src="https://github.com/Icaro92/MMxM/assets/58118599/1c493a59-60ac-479a-8de7-ac4d5b43d429" width="500" alt="SalesDash Image">

  * Using a Correlation Heatmap to confirm and quantify the influence of media channels over sales in the dataset
  * Interesting to notice that none of the media channels harm Sales. 
    
  <p align = "left">
  <img src="https://github.com/Icaro92/MMxM/assets/58118599/dd2001a4-5981-45f8-9d82-80675be10dc5" width="500" alt="SalesDash Image">

   **3 - Data Modeling**

   * The model was trained with a 20% split test and using a LinearRegression algorithm

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import numpy as np
    
    # Choosing Sales as the target variable in Labels and removing it from the Dataset (df)
    labels = pd.DataFrame(df, columns=['sales'])
    df = df.drop('sales', axis=1)
    
    #Train and Test Split. 20% for trainning. 
    train_df, test_df, train_labels, test_labels = train_test_split(df, labels, test_size=0.2, random_state=4)
  
    from sklearn.linear_model import LinearRegression #Importing Linear Regression Libary
    ln = LinearRegression()
    
    ln.fit(train_df, train_labels) #Training the model
    
    pred = ln.predict(test_df) #Applying the model in the Test dataset

  
   * The model provided a very good fit with a Coefficient of determination (R²_score) of 0.9, which indicates the model will provide correct results 90% of the time.

    #Verifying model Accuracy with R2_score
    from sklearn.metrics import r2_score
    r = r2_score(test_labels,pred)
    r_formatado = '{:.2f}'.format(r)
    print('r2_score:', r_formatado) # IMPORTANT: the model will provide the correct value 90% of the time.
     
   * I plotted a graph to compare the Real and Prediction to visualize the model accuracy

    # Plotting the Predictions (red) versus the real results (blue) in the test dataset.
    c = [i for i in range(1,36,1)]
    fig = plt.figure(figsize=(15,8))
    plt.plot(c,test_labels, color='blue', label='Real')
    plt.plot(c,pred,color='red', label='Prediction');
    plt.xlabel('Index')
    plt.ylabel('Sales')
    plt.legend();

<p align = "left">
    <img src="https://github.com/Icaro92/MMxM/assets/58118599/bf319d01-4039-4e87-af34-f4f330ac9173" width="500" alt="SalesDash Image">
   

   **4 - Validating - Use Example**

   * To make the results even more intelligible I have created a small Use Example where I compared the prediction of the results vs. the real result from the dataset.
<p align = "left">
    <img src="https://github.com/Icaro92/MMxM/assets/58118599/0056a7a6-28d5-4445-9a62-787907603da9" width="500" alt="SalesDash Image">

    #youtube = 212.4
    #facebook = 40.08
    #newspaper = 46.44
    input_data = (212.4,40.08,46.44)
    imput_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = imput_data_as_numpy_array.reshape(1,-1)
    sales_f = ln.predict(input_data_reshaped)[0]
    #sales_formatado = '{:.2f}'.format(sales_f)
    print('The Predicted Sales value will be: ',sales_f, 'And the real Sales value is 20.52')
    
    
