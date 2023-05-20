from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the Census Data model API! =)"}

def test_positive_inference():
    r = client.post(
        "/model/",
        json={
            'age':31,
            'workclass':'Private',
            'fnlgt':45781,
            'education':'Masters',
            'education-num':14,
            'marital-status':'Never-married',
            'occupation':'Prof-specialty',
            'relationship':'Not-in-family',
            'race':'White',
            'sex':'Female',
            'capital-gain':14084,
            'capital-loss':0,
            'hours-per-week':50,
            'native-country':'United-States'
        },
    )
    print(r.json)
    assert r.status_code == 200
    assert r.json() == {"pred": f"Models prediction is 1"}

def test_negative_inference():
    r = client.post(
        "/model/",
        json={
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
        },
    )
    print(r.json)
    assert r.status_code == 200
    assert r.json() == {"pred": f"Models prediction is 0"}
