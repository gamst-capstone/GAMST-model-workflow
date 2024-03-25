from flask import Flask, request, make_response
from dotenv import load_dotenv
import json
import requests
import boto3
import os

from capstone_function import generateCaption

app = Flask(__name__)
sqs = boto3.client('sqs')

load_dotenv()

@app.route('/receive', methods=['POST'])
def receive_message():
    # get SNS message from request
    try:
        msg = json.loads(request.data)

        hdr = request.headers.get('x-amz-sns-message-type')

        # confirm for SNS subscription
        if hdr == 'SubscriptionConfirmation' and 'SubscribeURL' in msg:
            r = requests.get(msg['SubscribeURL'])

        # get message from SNS
        if hdr == 'Notification':
            msg_process(msg['Message'], msg['Timestamp'])
    except Exception as e:
        print(e)
        ret_msg = {
            'status': 'fail',
        }
        pass


    ret_msg = {
        'status': 'success',
    }
    if ret_msg['status'] == 'success':
        return make_response(json.dumps(ret_msg), 200)
    else:
        return make_response(json.dumps(ret_msg), 400)

def get_sqs_url():
    res = sqs.get_queue_url(QueueName=os.environ['SQS_QUEUE_NAME'])
    return res['QueueUrl']

def msg_process(msg, timestamp):
    # process SNS message
    parsed_json = json.loads(msg)

    # get message from SQS queue
    queue_url = get_sqs_url()
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20,
    )
    sqs_msg = response['Messages'][0]

    # take Video URL from SQS message
    video_url = sqs_msg['Body']
    generateCaption(video_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)