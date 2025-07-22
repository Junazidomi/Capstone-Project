import requests
import time
import torch
import os
import smbus
import numpy as np
import librosa
import joblib

# Definisikan token API dan header
API_TOKEN = "SI HITAM"
HEADERS = {
    "api-token": API_TOKEN
}

# Fungsi untuk mendapatkan semua pengguna
def get_all_users():
    url = "https://zetra.cloud/api/allUsers"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Gagal mengambil data pengguna. Kode status: {response.status_code}")
        return []

# Fungsi untuk memperbarui suhu pengguna
def update_user_temperature(user_id, suhu, objecti, Cry_status, data_suara):
    url = f"https://zetra.cloud/api/users/{user_id}"
    objecti= not objecti
    data = {
        "data": {
            "suhu": suhu,
            "baby": objecti,
            "face": Cry_status,
            "sound": data_suara
        }
    }
    
    response = requests.put(url, json=data, headers=HEADERS)
    if response.status_code == 200:
        print(f"Data pengguna {user_id} berhasil diperbarui.")
    else:
        print(f"Gagal memperbarui data pengguna {user_id}. Kode status: {response.status_code}")
        print("Respon:", response.json())

# Kelas MLX90614
class MLX90614:
    MLX90614_TA = 0x06
    MLX90614_TOBJ1 = 0x07

    def __init__(self, address=0x5a, bus=1, shared_data=None):
        self.address = address
        self.bus = smbus.SMBus(bus)
        self.shared_data = shared_data

    def readValue(self, registerAddress):
        error = None
        for i in range(3):
            try:
                return self.bus.read_word_data(self.address, registerAddress)
            except IOError as e:
                error = e
                time.sleep(0.1)
        raise error

    def valueToCelsius(self, value):
        return -273.15 + (value * 0.02)

    def readObjectTemperature(self):
        value = self.readValue(self.MLX90614_TOBJ1)
        temperature = self.valueToCelsius(value)
        suhu = round(temperature, 2)
        formatted_temperature = f"{temperature:.2f}"
        print("Suhu Objek:", formatted_temperature, "C")

        self.shared_data['object_temp'] = suhu
        if self.shared_data['users']:
            for user in self.shared_data['users']:
                user_id = list(user.keys())[0]
                update_user_temperature(user_id, self.shared_data['object_temp'], self.shared_data['objecti'], self.shared_data['Cry_status'], self.shared_data['data_suara']) 
        return self.shared_data['object_temp']
class SoundClassifier:
    def __init__(self, model_path, folder_to_test, shared_data):
        self.model_path = model_path
        self.folder_to_test = folder_to_test
        self.model = self.load_model()
        self.shared_data = shared_data

    def extract_features(self, file_path, mfcc=True, chroma=True, mel=True):
        signal, sr = librosa.load(file_path, sr=None)
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
            features.extend(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T, axis=0)
            features.extend(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=0)
            features.extend(mel)
        return features

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            print("Error loading the model:", str(e))
            return None

    def predict_single_file_with_delay(self, file_path):
        features = self.extract_features(file_path)
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        sound_state = prediction == 'Cry'
        print('Suara', sound_state)
        state = 'Recorded'
        with open('/home/fahmi/yolov5/Proses/File/Audio.txt', 'w') as file:
            file.write(str(state))
        os.remove(file_path)
        return sound_state

    def process_files(self):
        if self.model is None:
            return None

        while True:
            for file in os.listdir(self.folder_to_test):
                file_path = os.path.join(self.folder_to_test, file)
                if file.endswith(".wav"):
                    self.shared_data['data_suara'] = self.predict_single_file_with_delay(file_path)
                    if self.shared_data['users']:
                        for user in self.shared_data['users']:
                            user_id = list(user.keys())[0]
                            update_user_temperature(user_id, self.shared_data['object_temp'], self.shared_data['objecti'], self.shared_data['Cry_status'], self.shared_data['data_suara'])
                    return self.shared_data['data_suara']

# Kelas ObjectDetection
class ObjectDetection:
    def __init__(self, object_weights, emotion_weights, folder_path, shared_data, img_size=640, conf_thresh=0.25):
        self.object_weights = object_weights
        self.emotion_weights = emotion_weights
        self.folder_path = folder_path
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.shared_data = shared_data
        self.object_model = torch.hub.load('ultralytics/yolov5', 'custom', path=object_weights)
        self.emotion_model = torch.hub.load('ultralytics/yolov5', 'custom', path=emotion_weights)

    def detect_objects_in_folder(self):
        files = os.listdir(self.folder_path)
        
        for file in files:
            if not file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(self.folder_path, file)
            if not os.path.exists(img_path):
                print(f"File {img_path} tidak ditemukan.")
                continue
            
            results = self.object_model(img_path, size=self.img_size)
            detected_objects = results.xyxy[0]
            detected_objects = detected_objects[detected_objects[:, 4] > self.conf_thresh]
            class_names = self.object_model.module.names if hasattr(self.object_model, 'module') else self.object_model.names
            class_predictions = [class_names[int(obj[5])] for obj in detected_objects]
            
            objecti = False
            Cry_status = True
            if 'Baby' in class_predictions and 'Other' in class_predictions:
                baby_idx = class_predictions.index('Baby')
                other_idx = class_predictions.index('Other')
                baby_conf = detected_objects[baby_idx, 4]
                other_conf = detected_objects[other_idx, 4]
                objecti = baby_conf > other_conf
            elif 'Baby' in class_predictions and 'Figure' in class_predictions:
                baby_idx = class_predictions.index('Baby')
                fig_idx = class_predictions.index('Figure')
                baby_conf = detected_objects[baby_idx, 4]
                fig_conf = detected_objects[fig_idx, 4]
                objecti = baby_conf > fig_conf
            elif 'Baby' in class_predictions:
                objecti = True
            elif 'Other' in class_predictions or 'Figure' in class_predictions:
                objecti = False
            else:
                objecti = False
            if objecti:
                Cry_status = self.emotion_detect(img_path)
            else:
                cap = 'Capture'
                Cry_status = False
                with open('/home/fahmi/yolov5/Proses/File/Image.txt', 'w') as file:
                    file.write(str(cap))
                os.remove(img_path)
            print('Baby?', objecti)
            print('Nangis?', Cry_status)
            self.shared_data['objecti'] = objecti
            self.shared_data['Cry_status'] = Cry_status
            if self.shared_data['users']:
                for user in self.shared_data['users']:
                    user_id = list(user.keys())[0]
                    update_user_temperature(user_id, self.shared_data['object_temp'],self.shared_data['objecti'],self.shared_data['Cry_status'], self.shared_data['data_suara'])
            time.sleep(0.5)
            
            
            return self.shared_data['objecti'],self.shared_data['Cry_status']

    def emotion_detect(self, img_path):
        results = self.emotion_model(img_path, size=self.img_size)
        detected_objects = results.xyxy[0]
        detected_objects = detected_objects[detected_objects[:, 4] > self.conf_thresh]
        class_names = self.emotion_model.module.names if hasattr(self.emotion_model, 'module') else self.emotion_model.names
        
        Emot = False
        class_predictions = [class_names[int(obj[5])] for obj in detected_objects]
        if 'Cry' in class_predictions and 'Sleep' in class_predictions:
            cry_idx = class_predictions.index('Cry')
            sleep_idx = class_predictions.index('Sleep')
            cry_conf = detected_objects[cry_idx, 4]
            sleep_conf = detected_objects[sleep_idx, 4]
            Emot = cry_conf > sleep_conf
        elif 'Cry' in class_predictions and 'Normal' in class_predictions:
            cry_idx = class_predictions.index('Cry')
            normal_idx = class_predictions.index('Normal')
            cry_conf = detected_objects[cry_idx, 4]
            normal_conf = detected_objects[normal_idx, 4]
            Emot = cry_conf > normal_conf
        elif 'Sleep' in class_predictions and 'Normal' in class_predictions:
            Emot = False
        elif 'Normal' in class_predictions or 'Sleep' in class_predictions:
            Emot = False
        elif 'Cry' in class_predictions:
            Emot = True		
        cap = 'Capture'
        with open('/home/fahmi/yolov5/Proses/File/Image.txt', 'w') as file:
            file.write(str(cap))
        os.remove(img_path)
        time.sleep(0.5)
        return Emot



# Fungsi utama untuk menjalankan skrip
if __name__ == "__main__":
    users = get_all_users()

    # Struktur data bersama
    shared_data = {
        'users': users,
        'object_temp': 21,
        'objecti': True,
        'Cry_status': True,
        'data_suara': False
    }

    for user in users:
        user_id = list(user.keys())[0]
        update_user_temperature(user_id, shared_data['object_temp'], shared_data['objecti'], shared_data['Cry_status'], shared_data['data_suara'])

    sensor = MLX90614(shared_data=shared_data)
    classifier = SoundClassifier(
        "/home/fahmi/Downloads/Sound_Model_2_Second.pkl", 
        "/home/fahmi/yolov5/Sound", 
        shared_data=shared_data
    )
    picdec = ObjectDetection(
        object_weights='/home/fahmi/yolov5/Baby_Version_2.pt',
        emotion_weights='/home/fahmi/yolov5/Emotion_FIX_640.pt',
        folder_path='/home/fahmi/yolov5/Picture',
        shared_data=shared_data
    )
    
    while True:
        shared_data['object_temp'] = sensor.readObjectTemperature()
        shared_data['data_suara'] = classifier.process_files()
        shared_data['objecti'], shared_data['Cry_status'] = picdec.detect_objects_in_folder()
        
