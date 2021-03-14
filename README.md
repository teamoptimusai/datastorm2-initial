# Final Implementation

 - Multi-Model Experiment
 - How to Run

### Multi-Model Experiment
In Deep Learning, we sometimes use a method called Multi-Model Ensembling where we combine pertained huge neural networks parallelly and we concatenate the feature vectors coming from those Networks and then use a simple classifier (could be an ANN or a Tree-based classifier) on the concatenated feature vector for classification. In researches such as Image Classification, this method has been proven to work very well.

So we used the same principle and made a multi-model classifier but by replacing Tree-based classifiers instead of Deep Neural Networks. As tree-based classifiers cannot produce feature vectors as outputs we have to just stick with their current outputs.

  

For this experiment, we divided the dataset into 3 parts namely (train1, train2, train3) and them as follows.

![enter image description here](https://i.ibb.co/RDsst9F/Data-Storm2-0-Initial-Round-Report-2.png)

Following are the Results for the Validation Set

    Accuracy - 0.49108766824299743
    Confusion Matrix - 
     [[1174  276  160]
     [ 529  146   66]
     [ 301   67   30]]
                  precision    recall  f1-score   support
    
               0       0.59      0.73      0.65      1610
               1       0.30      0.20      0.24       741
               2       0.12      0.08      0.09       398
    
        accuracy                           0.49      2749
       macro avg       0.33      0.33      0.33      2749
    weighted avg       0.44      0.49      0.46      2749

For Test Set we were able to get 0.33199 F1 Score. Which places us in Top 10 in the DataStorm 2.0 Kaggle Competition

### How to Run
clone the git repository to your computer and move to the relevant folder using the following comands

    $ git clone https://github.com/teamoptimusai/datastorm2-initial.git
    $ cd datastorm2-initial/final_implementation
If you want to tryout the current model just run the following model

    $ python main.py
If you want to tryout every possible models 

    $ python model_rotator.py

