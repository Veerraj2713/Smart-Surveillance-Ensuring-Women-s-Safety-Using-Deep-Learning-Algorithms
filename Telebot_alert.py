import telebot
import cv2
import time

# Initialize the Telegram bot
# bot = telebot.TeleBot("7740995810:AAFTiceaYwU6YmL3QLxrBHf5kYd7MNesIkI")
# CHAT_ID = "7637004353"

bot = telebot.TeleBot("7767843034:AAF4xNZ1l6amd769E-MTklAY9gLfWYV6JDU")
CHAT_ID = "1452885164"

last_alert_time = 0

def send_telegram_alert(frame, message):
    print("telegram alert started")
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time >= 60:
        try:
            cv2.imwrite("alert.jpg", frame)
            with open("alert.jpg", 'rb') as photo:
                bot.send_photo(CHAT_ID, photo, caption=f"🚨 ALERT! 🚨\n{message}")
            bot.send_message(CHAT_ID, f"{message} Please take necessary precautions immediately!")
            last_alert_time = current_time
            print(f"Telegram alert sent: {message}")
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
    else:
        print("Waiting to send next alert. Time since last alert:", int(current_time - last_alert_time), "seconds")
