# Census Project

Repo of code for project 3 of the MLDevOps course.
github repo url: https://github.com/claudia-fidelis/nd0821-c3-starter-code/edit/master

The target for this project is if a respondent's salary is greater than $50k or not.

This project trains a model and checks the performance of the data on slices of the dataset to understand fairness and bias.

The model has an api via FastAPI for inference, and the project includes code for deployment to Render.


### Environment Set up

Download and install conda if you don’t have it already.
Use the supplied requirements file to create a new environment, or
* `conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge`
* Install requirements: `pip install -r requirements.txt`


### Model

Call train_model.py script to train the model: 
`python starter/starter/train_model.py`


### Tests

Tests for machine learning can be found in nd0821-c3-starter-code/starter/starter/ml/test_model.py.
Tests for FastAPI implementation can be found in nd0821-c3-starter-code/starter/test_main.py.


### Continuous Integration

This is handled by GitHub Actions, the workflow: nd0821-c3-starter-code/.github/workflow/python-app.
The action uses flake8 to lint and runs the test scripts detailed in the Tests section above.


### Rest API
To startup the app local: 
`uvicorn main:app --reload`
URL: http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs


### Continuous Delivery
CD was made using Render and is triggered by pushes to the main branch. 
The new app will automatic be deployed to render: https://render-deployment-nd0821-c3.onrender.com/.

We have the post_request.py file, which can be executed on the command line to demonstrate the POST request.
