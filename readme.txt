Purpose

Prediction of the 10-day runoff risk level [0: No; 1: Low; 2: Medium and 3: High] over the Great Lakes region shown below. The region is defined by five clusters given their hydrologic characteristics. For each cluster, we trained an XGBoost model to predict the risk level and then the results are combined for the entire region. In the following, we will explain the workflow and its implementation with Python script.
 

Figure 1. The Great Lakes region defined by five clusters

Code description and workflow


The code workflow:

Data and model are the inputs to code in (1) & (2). 
Set all the paths inside the code at (1) and (2) file correctly. 
Then Run alldomain.py and var.py (no running order sequence). 
Output from (1 & 2) will be input for (3).  
Then Run alldomain_convert.py (Make sure you have outputs from both of the code files (1 & 2 before running this file code).
This will produce the netcdf (file format) file, which can be later read and displayed as a map-like figure. 
display.py contains code to display the final result from (3). 

1. alldomain.py - This file contains the code to make predictions of the runoff magnitude, runoff event and its probability from the national water model output by using the trained statistical model. Later it's output will be used by file in (3). 
 
Run Instructions:
    First Change the directory path wherever it is present according to your system path, where you have saved the files and where you want to save the results. 
    
    All Paths in the code and important instructions are commented. 
    [Hints: See the matching path with its file name and based on that change accordingly]
    
    Then run the code! 
2. var.py - This file contains code to get data from National Water Model variables namely QSNOW, ACSNOM and ACCPRCP (Rainrate). Later the output from this file is used in the file at (3).

    Follow Instructions at (1) to run this code. It is exactly similar to it! 


3. alldomain_convert.py - This file contains code to get the final overall outputs. 

Follow the instructions to change path in (1), similar to it and run it. 

The output from file (1) together with this file will create 4 main outputs:
Risk level for 10 days
Minimum Risk Level over 10 days
Maximum Risk Level over 10 days
Median Risk Level over 10 days. 

The output from file (2) together with this file will generate:
    
a) Maximum, Minimum and Median value for all the 3 variables from the National Water Model mentioned above in (2). 
    
    
4. data - This folder contains the data from the National Water Model that needs to be used 
by code in (1) and (2). It contains 10 days of data. 

6. model - This folder contains statistical model prediction for 5 clusters, which will be used by code in file (1) and (2). 

7. display.py - It contains code just to display the result. 




