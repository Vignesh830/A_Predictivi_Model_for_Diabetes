# A_Predictivi_Model_for_Diabetes
We developed a predictive model for diabetes detection using an SVM with a linear kernel. After loading the data and splitting it into training and testing sets, we used the SVM algorithm to train the model. We evaluated the model's performance on both the training and testing sets, and created several plots to gain a visual understanding of the data. This analysis and visualization gave us valuable insights into the factors influencing diabetes prediction.

So we executed this Project in Colab Notebook.
this code is python oriented

System Design
    The diabetes prediction project can be divided into two phases: data preparation and model training and evaluation.
    Phase 1: Data Preparation 
                 In the data preparation phase, the PIMA dataset is loaded and preprocessed. This includes handling missing values, scaling features, and converting categorical features to numerical features. The data is then split into two sets: a training set and a test set. The training set is used to train the SVM model, and the test set is used to evaluate the model's performance.
                 ![image](https://github.com/Vignesh830/A_Predictivi_Model_for_Diabetes/assets/159744719/af8f34d8-34b3-4c5c-ad36-ea4f4cf6b39b)
    Phase 2: Model Training and Evaluation: 
                 In the model training and evaluation phase, the SVM model is trained on the training set. The SVM algorithm finds a hyperplane in the feature space that separates the data points into two                   classes: diabetic and non-diabetic. Once the model is trained, it is evaluated on the test set to assess its performance. If the model's performance on the test set is satisfactory, it is saved                  to disk for future use.
                 Here the Final Result:
                 (768, 8) (614, 8) (154, 8) 
                 Accuracy Score of training data : 78.33876221498372 % 
                 Accuracy Score of testing data : 77.27272727272727 %
                 ![image](https://github.com/Vignesh830/A_Predictivi_Model_for_Diabetes/assets/159744719/7ff12a6e-67ff-48fe-a9a6-ddf51d195164)
                 ![image](https://github.com/Vignesh830/A_Predictivi_Model_for_Diabetes/assets/159744719/a3dee699-b112-44c1-a891-b7890ca26335)
                 ![image](https://github.com/Vignesh830/A_Predictivi_Model_for_Diabetes/assets/159744719/e055c0d3-89dc-48ac-bcab-a2fc2a7fcdaf)



