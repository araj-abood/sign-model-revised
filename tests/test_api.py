import requests

url = 'http://127.0.0.1:5000/predict'
files = {'image': open('test_image.jpg', 'rb')}  # Replace with a valid test image
response = requests.post(url, files=files)

print(response.json())