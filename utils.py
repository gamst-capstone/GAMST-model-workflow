import logging

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