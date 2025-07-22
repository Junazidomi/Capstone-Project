import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import numpy as np
import time
import os

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def record_audio_segment(filename, duration=2, fs=44100, cutoff=2000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Tunggu hingga rekaman selesai
    
    # Terapkan high-pass filter
    audio = audio.flatten()  # Flatten array untuk diproses
    filtered_audio = highpass_filter(audio, cutoff, fs)
    
    # Konversi kembali ke bentuk 2D array untuk penyimpanan
    filtered_audio = filtered_audio.astype(np.int16).reshape(-1, 1)
    
    # Simpan rekaman ke file
    write(filename, fs, filtered_audio)

def main():
    duration = 2  # Durasi setiap rekaman dalam detik
    fs = 44100  # Frekuensi sampling
    cutoff = 1000  # Frekuensi cutoff untuk high-pass filter
    folder_name = "/home/fahmi/yolov5/Sound"
    instruction_file = "/home/fahmi/yolov5/Proses/File/Audio.txt"  # File instruksi
    
    # Buat folder jika belum ada
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Rekam 2 detik pertama
    timestamp = int(time.time())
    filename = os.path.join(folder_name, f"rekaman_{timestamp}.wav")
    record_audio_segment(filename, duration, fs, cutoff)

    while True:
        with open(instruction_file, 'r') as file:
            Status = file.read().strip()
        
        if Status == "Recorded":
            timestamp = int(time.time())
            filename = os.path.join(folder_name, f"rekaman_{timestamp}.wav")
            record_audio_segment(filename, duration, fs, cutoff)
            
            # Reset file instruksi
            with open(instruction_file, 'w') as file:
                file.write("")
        
        time.sleep(duration)  # Tunggu selama durasi rekaman sebelum cek lagi

if __name__ == "__main__":
    main()
