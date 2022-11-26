import os
import boto3
import json
import urllib
from urllib.request import urlopen
from PIL import Image
import numpy as np

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']

def lambda_handler(event, context):
    data = event['body']
    resp_obj = urllib.request.urlopen(data)
    raw_img = Image.open(resp_obj)
    raw_img = raw_img.resize((28,28))
    np_array_img = np.array(raw_img)
    image = np_array_img.reshape((-1,28,28,1))
    np_img = np.array(image)
    input= json.dumps(np_img.tolist())
    runtime= boto3.client('runtime.sagemaker')

    sage_inp = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=input)
    
    result = json.loads(sage_inp['Body'].read().decode())
    
    predictions = result['predictions']
    image_arr = []
    
    for i in predictions:
        image_arr.append(max(i))
    final_prediction = max(image_arr)
    print(sage_inp)
    return final_prediction
    
    
    