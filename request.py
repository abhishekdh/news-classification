import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'headline':'MS Dhoni at No. 3 would have broken most records Gautam Gambhir',})
print(r.json())