import subprocess
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from time import sleep
import os
import time
import threading

# Menetapkan konstanta kamera
STREAM_RESOLUTION = (640, 480)  # Resolusi streaming
CAPTURE_RESOLUTION = (1920, 1080)  # Resolusi penangkapan
FRAMERATE = 24  # Sesuaikan dengan kemampuan kamera Anda

# URL dan Kunci Streaming YouTube
YOUTUBE_URL = "rtmp://zetra.cloud/live"
YOUTUBE_KEY = "obs_stream"  # Ganti dengan kunci streaming YouTube Anda

# Membuat perintah FFmpeg untuk streaming
stream_cmd = f'ffmpeg -ar 44100 -ac 2 -acodec pcm_s16le -f s16le -ac 2 -i /dev/zero -f h264 -i - -vcodec copy -acodec aac -ab 128k -g 50 -strict experimental -f flv {YOUTUBE_URL}/{YOUTUBE_KEY}'

# Inisialisasi kamera
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": STREAM_RESOLUTION, "format": "RGB888"})
picam2.configure(camera_config)

encoder = H264Encoder(bitrate=10000000)

# Memulai subprocess streaming
stream_pipe = subprocess.Popen(stream_cmd, shell=True, stdin=subprocess.PIPE)

# Fungsi untuk menangani streaming langsung
def start_streaming():
    output = FileOutput(stream_pipe.stdin)
    picam2.start_recording(encoder, output)

# Fungsi untuk menangani pengambilan gambar
def capture_images():
    # Direktori untuk menyimpan gambar
    directory = "/home/fahmi/yolov5/Picture"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Pengambilan gambar awal
    filename = os.path.join(directory, f"test_{int(time.time())}.jpg")
    picam2.capture_file(filename)

    while True:
        # Membaca status dari file
        try:
            with open("/home/fahmi/yolov5/Proses/File/Image.txt", "r") as file:
                status = file.read().strip()
        except FileNotFoundError:
            status = "Wait"  # Default ke "wait" jika file tidak ada

        if status.strip() == "Capture":
            filename = os.path.join(directory, f"test_{int(time.time())}.jpg")
            picam2.capture_file(filename)
            with open("/home/fahmi/yolov5/Proses/File/Image.txt", 'w') as file:
                file.write("")

# Memulai kamera
picam2.start()

sleep(2)  # Waktu pemanasan kamera

# Memulai thread streaming
streaming_thread = threading.Thread(target=start_streaming)
streaming_thread.start()

# Memulai thread pengambilan gambar
capturing_thread = threading.Thread(target=capture_images)
capturing_thread.start()

try:
    while True:
        sleep(10)
except KeyboardInterrupt:
    # Menghentikan perekaman saat menerima interupsi keyboard (Ctrl+C)
    picam2.stop_recording()
    stream_pipe.stdin.close()
    stream_pipe.wait()
    picam2.close()
