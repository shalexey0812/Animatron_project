import os
import shutil
import cv2
import requests
import time
from datetime import datetime, timedelta
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import get_checkpoint_url

# Ensure the photos directory exists
photos_dir = "Animatron_photos"
os.makedirs(photos_dir, exist_ok=True)

# Function to clear the photos directory every week
def clear_photos_directory():
    global last_clear_time
    if datetime.now() - last_clear_time > timedelta(weeks=1):
        for filename in os.listdir(photos_dir):
            file_path = os.path.join(photos_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        last_clear_time = datetime.now()

# Define a class to handle the Detectron2 model operations
class DetectronModel:
    def __init__(self, config_path, weights_url, threshold=0.5, animal_ids=None):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.MODEL.WEIGHTS = weights_url
        self.animal_ids = animal_ids if animal_ids else []
        self.predictor = DefaultPredictor(self.cfg)
        self.last_alert_time = 0

    def process_frame(self, frame, bot):
        outputs = self.predictor(frame)
        pred_classes = outputs["instances"].pred_classes if outputs["instances"].has("pred_classes") else None
        animals_detected = False

        if pred_classes is not None:
            for class_id in pred_classes:
                if class_id.item() in self.animal_ids:
                    animals_detected = True
                    break

        current_time = time.time()
        if animals_detected and (current_time - self.last_alert_time) > 5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = os.path.join(photos_dir, f"detected_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            bot.send_message("Alert! There are unwanted species in the chicken coop!")
            bot.send_photo(frame_path)
            self.last_alert_time = current_time

# Define a class to handle Telegram bot operations
class TelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, message):
        try:
            url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
            data = {'chat_id': self.chat_id, 'text': message}
            response = requests.post(url, data=data)
            print(response.json())  # For debugging
        except Exception as e:
            print(f"An error occurred while sending message: {e}")

    def send_photo(self, photo_path):
        try:
            url = f'https://api.telegram.org/bot{self.bot_token}/sendPhoto'
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id}
                requests.post(url, files=files, data=data)
        except Exception as e:
            print(f"An error occurred while sending photo: {e}")

# Initialize the Detectron2 model
setup_logger()
animal_ids = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]
config_path = "C:/Users/alexs/coder/shaitan/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
weights_url = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
model = DetectronModel(config_path, weights_url, threshold=0.5, animal_ids=animal_ids)

# Initialize the Telegram bot
bot = TelegramBot(bot_token="7053063884:AAG5LQPQ1OcCSrXneatH05032U2SUv746H0", chat_id="609074184")

# Function to check for start/stop commands from the Telegram bot
def check_bot_commands(bot):
    global send_alerts
    try:
        url = f'https://api.telegram.org/bot{bot.bot_token}/getUpdates'
        response = requests.get(url).json()
        if response["ok"]:
            for update in response["result"]:
                if "message" in update and "text" in update["message"]:
                    message_text = update["message"]["text"]
                    if message_text == "/start":
                        send_alerts = True
                    elif message_text == "/stop":
                        send_alerts = False
                    # Acknowledge the update to not process it again
                    last_update_id = update["update_id"]
                    requests.get(f'https://api.telegram.org/bot{bot.bot_token}/getUpdates?offset={last_update_id+1}')
    except Exception as e:
        print(f"An error occurred while checking bot commands: {e}")

# Initialize video capture and start the main loop
cap = cv2.VideoCapture(0)
send_alerts = True
last_clear_time = datetime.now()  # Initialize the last clear time

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if send_alerts:
        model.process_frame(frame, bot)
    check_bot_commands(bot)
    clear_photos_directory()  # Call this function inside your main loop
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
