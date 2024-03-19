import os
import shutil
# these are two libraries to control the system.
import cv2
# arguably the most important library for any video processing in python.
import requests
# helps with system-internet connection.
import time
from datetime import datetime, timedelta
# libraries for time control.
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import get_checkpoint_url
# the model of Facebook Detectron2 itself.


# Initialize logging
log_file = "program_log.txt"

# Function to log messages to the log file
def log_message(message):
    # Open the log file and append the message with a timestamp
    with open(log_file, "a") as log:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"{timestamp} - {message}\n")

# Log the start of the program
log_message("Program started.")

# Ensure the photos directory exists
photos_dir = "Animatron_photos"
os.makedirs(photos_dir, exist_ok=True)
# Log the verification of the photos directory
log_message("Photos directory verified.")

# Function to clear the photos directory every week
def clear_photos_directory():
    global last_clear_time
    # Check if a week has passed since the last clear
    if datetime.now() - last_clear_time > timedelta(weeks=1):
        # Iterate through each file in the photos directory
        for filename in os.listdir(photos_dir):
            file_path = os.path.join(photos_dir, filename)
            try:
                # Delete the file or directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                # Log the deletion of the file or directory
                log_message(f"Deleted {file_path}.")
            except Exception as e:
                # Log any errors encountered during deletion
                log_message(f"Failed to delete {file_path}. Reason: {e}")
        # Update the last clear time to now
        last_clear_time = datetime.now()
        # Log the clearing of the directory
        log_message("Photos directory cleared.")

# Define a class to handle the Detectron2 model operations
class DetectronModel:
    def __init__(self, config_path, weights_url, threshold=0.9, animal_ids=None):
        # Initialize the model configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.MODEL.WEIGHTS = weights_url
        self.animal_ids = animal_ids if animal_ids else []
        self.predictor = DefaultPredictor(self.cfg)
        self.last_alert_time = 0
        # Log the initialization of the model
        log_message("Detectron2 model initialized with config: " + config_path)

    def process_frame(self, frame, bot):
        # Process the frame to detect animals
        outputs = self.predictor(frame)
        pred_classes = outputs["instances"].pred_classes if outputs["instances"].has("pred_classes") else None
        animals_detected = False

        # Check if any detected class is in the list of animal IDs
        if pred_classes is not None:
            for class_id in pred_classes:
                if class_id.item() in self.animal_ids:
                    animals_detected = True
                    break

        # Send a notification if animals are detected and it's been more than 5 seconds since the last notification
        current_time = time.time()
        if animals_detected and (current_time - self.last_alert_time) > 5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = os.path.join(photos_dir, f"detected_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            bot.send_message("Alert! There are unwanted species in the chicken coop!")
            bot.send_photo(frame_path)
            self.last_alert_time = current_time
            # Log the detection and notification
            log_message(f"Detected animals and notified at {timestamp}.")

# Define a class to handle Telegram bot operations
class TelegramBot:
    def __init__(self, bot_token, chat_id):
        # Initialize the Telegram bot with the provided token and chat ID
        self.bot_token = bot_token
        self.chat_id = chat_id
        # Log the initialization of the Telegram bot
        log_message("Telegram bot initialized.")

    def send_message(self, message):
        # Send a text message to the chat
        try:
            url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
            data = {'chat_id': self.chat_id, 'text': message}
            response = requests.post(url, data=data)
            # Log the sending of the message
            log_message(f"Message sent: {message}")
        except Exception as e:
            # Log any errors encountered when sending the message
            log_message(f"An error occurred while sending message: {e}")

    def send_photo(self, photo_path):
        # Send a photo to the chat
        try:
            url = f'https://api.telegram.org/bot{self.bot_token}/sendPhoto'
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': self.chat_id}
                requests.post(url, files=files, data=data)
            # Log the sending of the photo
            log_message(f"Photo sent: {photo_path}")
        except Exception as e:
            # Log any errors encountered when sending the photo
            log_message(f"An error occurred while sending photo: {e}")

# Initialize the Detectron2 model with the specified configuration and weights
setup_logger()
animal_ids = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]
config_path = "C:/Users/alexs/coder/shaitan/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
weights_url = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
model = DetectronModel(config_path, weights_url, threshold=0.9, animal_ids=animal_ids)
# Log the completion of model and Telegram bot initialization
log_message("Model and Telegram bot are initialized.")

# Initialize the Telegram bot with the specified bot token and chat ID
bot = TelegramBot(bot_token="some_token", chat_id="some_chat_id")

# Function to check for start/stop commands from the Telegram bot
def check_bot_commands(bot):
    global send_alerts
    # Attempt to retrieve and process commands from the Telegram bot
    try:
        url = f'https://api.telegram.org/bot{bot.bot_token}/getUpdates'
        response = requests.get(url).json()
        if response["ok"]:
            # Iterate through each update received from the bot
            for update in response["result"]:
                if "message" in update and "text" in update["message"]:
                    message_text = update["message"]["text"]
                    # Enable alerts if the start command is received
                    if message_text == "/start":
                        send_alerts = True
                        # Log the reception of the start command
                        log_message("Received start command.")
                    # Disable alerts if the stop command is received
                    elif message_text == "/stop":
                        send_alerts = False
                        # Log the reception of the stop command
                        log_message("Received stop command.")
                    # Acknowledge the update to prevent it from being processed again
                    last_update_id = update["update_id"]
                    requests.get(f'https://api.telegram.org/bot{bot.bot_token}/getUpdates?offset={last_update_id+1}')
    except Exception as e:
        # Log any errors encountered while checking for bot commands
        log_message(f"An error occurred while checking bot commands: {e}")

# Initialize video capture for the main program loop
cap = cv2.VideoCapture(0)
# Initially allow alerts to be sent
send_alerts = True
# Set the last time the photos directory was cleared to the current time
last_clear_time = datetime.now()

# Main program loop
while True:
    # Attempt to read a frame from the video capture
    ret, frame = cap.read()
    # Break from the loop if no frame is captured
    if not ret:
        break
    # If alerts are enabled, process the captured frame
    if send_alerts:
        model.process_frame(frame, bot)
    # Check for any commands received from the Telegram bot
    check_bot_commands(bot)
    # Attempt to clear the photos directory based on the set interval
    clear_photos_directory()
    # Display the captured frame in a window
    cv2.imshow("Frame", frame)
    # Allow the program to be exited by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Log the user-initiated termination of the program
        log_message("Program terminated by user.")
        break

# Release the video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
# Log the end of the program
log_message("Program ended.")

