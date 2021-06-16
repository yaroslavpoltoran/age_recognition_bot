import telebot
import urllib
import numpy as np
import face_crop
import config
import cv2
from PIL import Image


TOKEN = config.TOKEN
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, отправь мне фото :)')


@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text.lower() == 'привет':
        bot.send_message(message.chat.id, 'Привет!')
    elif message.text.lower() == 'пока':
        bot.send_message(message.chat.id, 'Прощай!')


@bot.message_handler(content_types=['photo'])
def photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    req = urllib.request.urlopen(f'https://api.telegram.org/file/bot{TOKEN}/{file_info.file_path}')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    photo_with_age, num_faces = face_crop.crop_face(img)
    if num_faces == 0:
        bot.send_message(message.chat.id, 'Ни одного лица не распознано',
                         reply_to_message_id=message.message_id)
    else:
        photo_with_age_img = Image.fromarray(photo_with_age)
        bot.send_photo(message.chat.id, photo_with_age_img,
                       reply_to_message_id=message.message_id,
                       caption=f'Распознано лиц: {num_faces}')


bot.polling()
