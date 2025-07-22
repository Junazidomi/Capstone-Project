import subprocess

def run_program_1():
    subprocess.run(['python', 'program_1.py'])

def run_program_2():
    subprocess.run(['python', 'program_2.py'])

def run_program_3():
    subprocess.run(['python', 'program_3.py'])



if __name__ == "__main__":
    # Jalankan kedua program secara bersamaan menggunakan subprocess
    process_1 = subprocess.Popen(['python', '/home/fahmi/yolov5/Proses/Recorded.py'])
    process_2 = subprocess.Popen(['python', '/home/fahmi/yolov5/Proses/Video_process'])
    process_3 = subprocess.Popen(['python', '/home/fahmi/yolov5/CGS/Triel/PYTORCH_INFERENCE/pkl.py'])
    process_1.wait()
    process_2.wait()
    process_3.wait()


