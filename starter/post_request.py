import requests
import json

data = {
    'age':39,
    'workclass':'State-gov',
    'fnlgt':77516,
    'education':'Bachelors',
    'education-num':13,
    'marital-status':'Never-married',
    'occupation':'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital-gain':2174,
    'capital-loss':0,
    'hours-per-week':40,
    'native-country':'United-States'
    }

r = requests.post("http://127.0.0.1:8000/model/", data = json.dumps(data))

print('status code:', r.status_code)
print('salary:', r.json())
