from flask import Flask, request, make_response
from dotenv import load_dotenv
import json
import requests
import boto3
import os
import logging

from capstone_function import generateCaption

app = Flask(__name__)
load_dotenv()
sqs = boto3.client('sqs', region_name=os.environ['AWS_REGION'], aws_access_key_id=os.environ['AWS_ACCESS_KEY'], aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


@app.route('/receive', methods=['POST'])
async def receive_message():
    # get SNS message from request
    try:
        msg = json.loads(request.data)

        hdr = request.headers.get('x-amz-sns-message-type')

        # confirm for SNS subscription
        if hdr == 'SubscriptionConfirmation' and 'SubscribeURL' in msg:
            r = requests.get(msg['SubscribeURL'])

        # get message from SNS
        if hdr == 'Notification':
            await msg_process(msg['Message'], msg['Timestamp'])

        ret_msg = {
            'status': 'success',
        }
    except Exception as e:
        logger.error(e)
        ret_msg = {
            'status': 'fail',
        }
        pass

    if ret_msg['status'] == 'success':
        return make_response(json.dumps(ret_msg), 200)
    else:
        return make_response(json.dumps(ret_msg), 400)

def get_sqs_url():
    res = sqs.get_queue_url(QueueName=os.environ['SQS_QUEUE_NAME'])
    return res['QueueUrl']

async def msg_process(msg, timestamp):
    try:
        # process SNS message
        parsed_json = json.loads(msg)
        logger.info(parsed_json['body'])
        body = json.loads(parsed_json['body'])
        s3 = body['Records'][0]
        if s3.get('s3'):
            bucket_name = s3['s3']['bucket']['name']
            object_name = s3['s3']['object']['key']
            video_url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
            logger.info(video_url)

            await generateCaption(video_url)
            logger.info("model working")
    except Exception as e:
        logger.error(e)
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)