## "Empathy Prediction Modeling: Preprocessing, Modeling, and Evaluation" 
The project's foundation is the prediction of empathy levels using visual data and gaze type. In this study effort, 60 participants took part in an experiment that involved them doing multiple trials.
* Advice: The study is titled ["EyeT4Empathy: Dataset of Foraging for Visual Information, Gaze Typing, and Empathy Assessment."](https://www.nature.com/articles/s41597-022-01862-w.) You can read all the details of the experiment and the dataset alignment in the study article.
* Datasets: [EyeT4Empathy dataset](https://figshare.com/articles/dataset/Eye_Tracker_Data/19729636/2) - 60 participants' 502 files and their recordings.
    <br />  [Questionnaries](https://figshare.com/articles/dataset/Questionnaires/19657323/2) - The empathy scores we're anticipating are there in this data.

[Data Exploration.ipynb]() - used a single participant's trail file to investigate the data, preprocess it, and visualise it using several graph plots for a better understanding.
<br/>[Empathy.ipynb]() - Detailed code execution for this project is in this Jupyter Notebook, which includes Data Understanding, Data Preprocessing, Feature Selection, Model Training, and Model Evaluation. 
<br/>[main code.py]() - The most accurate model, which uses all the files for Data Preprocessing, Feature Selection, Model Training, and Model Evaluation, is coded in a Python file. 
