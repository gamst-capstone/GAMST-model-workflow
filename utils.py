import boto3, os

def video_id_check(conn, video_uid):
    with conn.cursor() as cursor:
        sql = f"SELECT id FROM video_video WHERE url LIKE %s"
        cursor.execute(sql, ('%' + video_uid + '%',))
        video_id = cursor.fetchone()[0]
        if not video_id:
            return []
        return video_id
    
def camera_id_check(conn, camera_id, camera_url):
    with conn.cursor() as cursor:
        sql = f"SELECT id FROM camera_camera WHERE stream_url LIKE %s"
        cursor.execute(sql, (camera_url,))
        res = cursor.fetchone()[0]
        if not res or (res != int(camera_id)):
            return []
        return camera_id
    
def upload_to_s3(bucket, filename):
    # check the video file is exists
    if not os.path.exists(filename):
        print(f"File not found : {filename}")
        return False

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )

    try:
        res = s3.upload_file(filename, bucket, filename)
        return {
            "status": True,
            "file_url": f"https://{bucket}.s3.amazonaws.com/{filename}"
        }
    except Exception as e:
        print(e)
        return False
