import yt_dlp

def download_video(url, output_path='./input_video/downloaded_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', 
        'outtmpl': output_path, 
        'merge_output_format': 'mp4', 
        'noplaylist': True, 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url]) 

video_url = 'https://youtu.be/AtqGmSMSvzU?si=4nU4r8LznaNQabQF' # Add your URL here
download_video(video_url)