from flask import Flask, request, make_response
from dotenv import load_dotenv
from datetime import datetime
import json
import requests
import boto3
import os
import logging
import threading

from capstone_function import generateCaption, loadCaptionModel, loadDetectionModel, loadSentimentalModel
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
load_dotenv()

sqs = boto3.client(
    'sqs',
    region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY", ""),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "")
)

logger.info("[*] Loading Models...")
loadCaptionModel()
loadDetectionModel()
loadSentimentalModel()

thread = threading.Event()

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
            # asyncio.create_task(msg_process(msg['Message'], msg['Timestamp']))
            thread.set()
            msg_process_thread = threading.Thread(target=msg_process, args=(msg['Message'], msg['Timestamp']))
            msg_process_thread.start()
            
            logger.info("msg_process started")

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


@app.route('/start-stream', methods=['POST'])
def start_stream():
    '''
    Parameter:
        {
            camera_id: int
            camera_name: string
            camera_url: string
            start_time: string
        }
    Return:
        {
            result: string[success, fail]
        }
    '''
    data = request.get_json()

    start_time = str(datetime.now())
    data['start_time'] = start_time

    try:
        thread.set()
        stream_thread = threading.Thread(target=stream_process, args=(data,))
        stream_thread.start()

        return make_response(json.dumps({'result': 'success'}), 200)
    except Exception as e:
        logger.error(e)
        return make_response(json.dumps({'result': 'fail', 'msg': str(e)}), 400)

def stream_process(data):
    input_object = {
        'type': 'stream',
        'id' : data.get('camera_id'),
        'url': data.get('camera_url'),
        'start_time': data.get('start_time'),
    }
    generateCaption(input_object)
    logger.info("[*] Model Working")

def msg_process(msg, timestamp):
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
            logger.info(f"Video URL : {video_url}")

            input_object = {
                'type': 'video',
                'url': video_url,
                'video_uid': video_url.split('/')[-1],
            }

            generateCaption(input_object)
            logger.info("[*] model working")
    except Exception as e:
        logger.error(e)
        pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)