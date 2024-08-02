from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import os
import asyncio
import logging
from inference import generate_response

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=os.environ["TOKEN"])
dp = Dispatcher()

async def send_welcome(message: types.Message):
    logger.info("Received /start command")
    await message.answer("Привет! Чтобы я мог ответить на твой вопрос, обратись ко мне по имени — «бот» прямо в тексте своего вопроса, иначе я не пойму у кого ты спрашиваешь. Для получения более подробной информации используй команду /help.")

async def send_help(message: types.Message):
    logger.info("Received /help command")
    await message.answer("Привет! Я бот-ассистент на базе квантизированной модели Saiga/Llama3 8B. Сейчас я ещё не совсем готов к работе, и чтобы стать умнее и полезнее, мне нужно дообучение на нужных диалогах в Telegram. Скрипт для этого дообучения находится в файле AI.py.")

async def echo_message(message: types.Message):
    logger.info(f"Received message: {message.text}")
    if "бот" in message.text.lower():  # Проверяем наличие слова "бот" в тексте сообщения
        await message.answer("Сейчас подумаю и скажу")  # Отправляем сообщение о том, что бот обдумывает ответ
        prompt = message.text  # Получаем текст сообщения
        response = generate_response(prompt)  # Генерируем ответ с помощью нейросети
        logger.info(f"Generated response: {response}")
        await message.answer(text=response)  # Отправляем ответ


def register_handlers(dp: Dispatcher):
    dp.message.register(send_welcome, Command(commands=['start']))
    dp.message.register(send_help, Command(commands=['help']))
    dp.message.register(echo_message)

async def main():
    logger.info("Starting bot")
    register_handlers(dp)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
