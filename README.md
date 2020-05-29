# Motor Maintenance Prediction
This is the code repository for Motor Maintenance Prediction project in Summer 2019.    
Please contact Ji-Eun Lee (jvlee@seas.upenn.edu) for questions.   

## Overview 
The Hershey Company has many factories that manufacture products. If a motor breaks down, this will cause immense loss. Can we create a model that sends out warnings requesting maintenance, when suspicious behavior is detected? This can prevent loss caused by sudden motor failure.   
In this project we build a model that can send different levels of warnings for maintenance request. Our model accomplish to predict imminent failure at least 13 hours prior to motor fault.  
For implementation details please refer to the technical report. 

## Dataset 
We use IMS Bearing data, a public dataset which can be downloaded from [this link](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing).    
To run this code repository, dataset has to be downloaded under *dataset/* directory.   

## Directories    
- src/  
- dataset/IMS/    
- results/  

## Sample commands 
### `src/preprocess_data.py`     
Create the representation vector for each one second recording, in dataset 2.   
`python src/preprocess_data.py --dataset 2`               
You can skip this process since the preprocessed data files are provided under dataset/IMS/.      

### `src/main.py`
Train using the first 20% of dataset 2 and test on the latter 80% of dataset 2.    
Save computed likelihood scores if save = 1.     
Using the trained model, test on new\_dataset 3.    
Save computed likelihood scores if save\_new = 1.      
`python src/main.py --dataset 2 --save 1 --new-testset 3 --save-new 1`     

### `src/visualize.py`
Load likelihood scores of model trained using the first 20% of dataset 2 and tested on the latter 80% of dataset 2.
Using raw drop values (delta), if {likelihood of t prediction} - {likelihood of t-1 prediction} < threshold -1000, send out warning.    
If warning is sent three time in a row, change to red alert.     
`python src/visualize.py --dataset 2 --threshold -1000`         
         
[Result]          
Warning 1 days 23:20:00 prior motor fault         
Warning 1 days 23:10:00 prior motor fault         
Warning 0 days 18:50:00 prior motor fault         
Warning 0 days 13:40:00 prior motor fault         
Warning 0 days 13:30:00 prior motor fault         
         
Predicted Imminent Failure 0 days 13:20:00 prior motor fault         

Load likelihood scores of model trained using the first 20% of dataset 3 and tested on dataset 2.    
Using percentage change, if {{likelihood of t prediction} - {likelihood of t-1 prediction}} / {likelihood of t-1 prediction} < threshold -0.05, send out warning.    
If warning is sent three time in a row, change to red alert.             
`python src/visualize.py --new-testset 2 --drop-metric % --threshold -0.05`         

[Result]   
Warning 4 days 04:50:00 prior motor fault   
Warning 1 days 23:20:00 prior motor fault   
Warning 0 days 05:50:00 prior motor fault   
Warning 0 days 04:20:00 prior motor fault   
Warning 0 days 04:10:00 prior motor fault   
   
Predicted Imminent Failure 0 days 04:00:00 prior motor fault            


