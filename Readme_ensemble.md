### Ensemble
The trained models are not saved and we cannot find the files we used to do the ensemble. So the code cannot  run. But we want to provide some details of how we ensemble multiple models. 

Please refer to the **ensemble** function in **run_ensemble.py** to see how we do the ensemble. It basically has 2 steps:
- Before doing the ensemble, we need to obtain the predicted scores from different models (they are not provided here).  
- Load the predicted scores and sum them up to get the final score which is finally used to do the evaluation.

