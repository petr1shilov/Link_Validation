from telegram.ext import Application, MessageHandler, filters, CommandHandler
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
import logging
import fitz

import re
from sentence_transformers import SentenceTransformer, util
import requests
import uuid
import json
import os.path


tresh_hold = 0.78
auth = 'Mzc3ODA5ODEtYjMxZC00N2MxLTliODktODEzM2ExYWFlNDA4OjkzMmQzZjE5LWQzYzktNGQ1YS04Y2NlLTZlZGE5OThhYzgzMQ=='
model = SentenceTransformer('intfloat/multilingual-e5-large')
TOKEN = '7048664683:AAEqHp4HfMWq96eaoQECnzkQy8CHplntAbM'


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def start(update, context):
    reply_keyboard = [["/start", "/help", "/launch_app", "/download", "/rerun"]]

    user = update.message.from_user
    logger.info("Старт %s: %s", user.first_name, update.message.text)

    await update.message.reply_text(
        "Привет. Меня зовут Reference Check Bot.\n"
        "Нажми кнопку /help что бы посмотреть как работает бот.\n",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=False, input_field_placeholder="Отправьте пдф и текст"
        )
    )

async def help(update, context):
    await update.message.reply_text(
        "Что бы начать работу нажмите кнопку - /start\n"
        "Необходимо скинкуть PDF файл с источком, немного подождать, пока происходит магия\n"
        "После чего можно скачать промеченные PDF файл нажав кнопку - /download\n"
        "Начать пайплайн заново - /rerun\n"
        "/launch_app\n"
        "Можно посмотеть инстукцию нажав - /help")



async def downloader(update, context):
    file = await context.bot.get_file(update.message.document)
    await file.download_to_drive('downloaded_file.pdf')
    if os.path.isfile('downloaded_file.pdf'):
        await update.message.reply_text('Файл получен')
    else:
         await update.message.reply_text('Файл не получен')

async def send_document(update, context):
    chat_id = update.message.chat_id
    document = open('modified_document.pdf', 'rb')
    await context.bot.send_document(chat_id, document)

async def rerun(update, context):
    if os.path.isfile("downloaded_file.pdf") \
            or os.path.isfile("modified_document.pdf") \
                or os.path.isfile("request_file.txt"):
        os.remove("downloaded_file.pdf")
        os.remove("modified_document.pdf")
        os.remove("request_file.txt")
    else:
        await update.message.reply_text('Загрузите новые документы')

def prepare_text(document):
    new_txt_list = []

    text = chr(12).join([page.get_text() for page in document])

    text = re.sub(r'(\.\d+,\d+)', '.', text)
    text = re.sub(r'([.]\d+)', '.', text)
    text = re.sub(r'(\.\–?\d+)', '.', text)
    text_new = re.split(r'(?=[.]\s[А-ЯA-Z])', text)

    for i in text_new:
        new_txt_list.append(re.sub(r'([.]\s)', '', i))

    for j in range(len(new_txt_list)):
        new_txt_list[j] = re.sub(r'([-]\n)', '', new_txt_list[j])

    return new_txt_list




def modifi_document(sentences, document):
    for page in document:
        for sentens in sentences:
            text_instances = page.search_for(sentens['text'])
            highlight = page.add_highlight_annot(text_instances)
            if sentens['color'] == 'green':
                highlight.set_colors(stroke=[0.8, 1, 0.8]) # light red color (r, g, b)
            else:
                highlight.set_colors(stroke=[1, 0.8, 0.8]) # light red color (r, g, b)
            highlight.update()
    document.save("modified_document.pdf")

async def echo(update, context):
    """Echo the user message."""
    
    if os.path.isfile("request_file.txt"):
        await update.message.reply_text('Запрос уже был отправлен')
    else:
        text = update.message.text
        my_file = open("request_file.txt", "w+")
        my_file.write(text)
        my_file.close()



def embeding(text_query, text_links, tresh_hold):
    """
      Отбор предложений-кандидатов

      Параметры:
      - text_query (str): запрос (текст для валидации)
      - text_links (list): список предложений из источника

      Возвращает:
      - словарь с парой ключ-знанение, где ключ - порядновый номер предложения-кандидата
            значние - само предложение-кандидат
      """
    
    list_of_candidates = []
    dict_of_all_candidats = {}

    text_links.append(text_query)
    
    # embeddings_query = model.encode(text_query, normalize_embeddings=False)
    embeddings_links = model.encode(text_links, normalize_embeddings=False)
    embeddings_query = embeddings_links[-1]
    embeddings_links = embeddings_links[: -1]

    answer = util.cos_sim(embeddings_query, embeddings_links)[0]

    text_links.pop()

    for i in range(len(answer)):
        dict_of_all_candidats[i] = {'text' : text_links[i], 'embedder_score' : float(answer[i])}
        if answer[i] > tresh_hold:
            list_of_candidates.append(i)

    return dict_of_all_candidats, list_of_candidates

def get_token(auth_token, scope='GIGACHAT_API_PERS'):
    """
      Выполняет POST-запрос к эндпоинту, который выдает токен.

      Параметры:
      - auth_token (str): токен авторизации, необходимый для запроса.
      - область (str): область действия запроса API. По умолчанию — «GIGACHAT_API_PERS».

      Возвращает:
      - ответ API, где токен и срок его "годности".
      """
    # Создадим идентификатор UUID (36 знаков)
    rq_uid = str(uuid.uuid4())

    # API URL
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    # Заголовки
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': rq_uid,
        'Authorization': f'Basic {auth_token}'
    }

    # Тело запроса
    payload = {
        'scope': scope
    }

    try:
        # Делаем POST запрос с отключенной SSL верификацией
        # (можно скачать сертификаты Минцифры, тогда отключать проверку не надо)
        response = requests.post(url, headers=headers, data=payload, verify=False)
        return response
    except requests.RequestException as e:
        print(f"Ошибка: {str(e)}")
        return -1
    

def get_chat_completion(auth_token, user_message):
    """
    Отправляет POST-запрос к API чата для получения ответа от модели GigaChat.

    Параметры:
    - auth_token (str): Токен для авторизации в API.
    - user_message (str): Сообщение от пользователя, для которого нужно получить ответ.

    Возвращает:
    - str: Ответ от API в виде текстовой строки.
    """
    # URL API, к которому мы обращаемся
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    # Подготовка данных запроса в формате JSON
    payload = json.dumps({
        "model": "GigaChat-Pro",  # Используемая модель
        "messages": [
            {
                "role": "user",  # Роль отправителя (пользователь)
                "content": user_message  # Содержание сообщения
            }
        ],
        "temperature": 1,  # Температура генерации
        "top_p": 0.1,  # Параметр top_p для контроля разнообразия ответов
        "n": 1,  # Количество возвращаемых ответов
        "stream": False,  # Потоковая ли передача ответов
        "max_tokens": 512,  # Максимальное количество токенов в ответе
        "repetition_penalty": 1,  # Штраф за повторения
        "update_interval": 0  # Интервал обновления (для потоковой передачи)
    })

    # Заголовки запроса
    headers = {
        'Content-Type': 'application/json',  # Тип содержимого - JSON
        'Accept': 'application/json',  # Принимаем ответ в формате JSON
        'Authorization': f'Bearer {auth_token}'  # Токен авторизации
    }

    # Выполнение POST-запроса и возвращение ответа
    try:
        response = requests.request("POST", url, headers=headers, data=payload, verify=False)
        return response
    except requests.RequestException as e:
        # Обработка исключения в случае ошибки запроса
        print(f"Произошла ошибка: {str(e)}")
        return -1



def answer(text_query, links, tresh_hold=tresh_hold):

    # text_links_prep = prepare_text(links)

    text_links_prep = links

    text_links, list_cand = embeding(text_query, text_links_prep, tresh_hold)

    response = get_token(auth)
    if response != -1:
        giga_token = response.json()['access_token']

    answer = []

    for i in text_links:
        text_n = text_links[i]['text']

        if i in list_cand:
            text_for_api = f'Подтверждается ли текст "{text_query}" текстом "{text_n}"\nОтветь только "да" или "нет"'
            answer_n = get_chat_completion(giga_token, text_for_api)
            llm_response = str(answer_n.json()['choices'][0]['message']['content']).lower()
            if 'да' in llm_response:
                answer.append({'sentence_idx' : i
                            ,'text' : text_n
                            ,'color' : 'green'
                            ,'embedder_score': text_links[i]['embedder_score']
                            ,'LLM_response' : llm_response}) 
            elif 'нет' in llm_response:
                answer.append({'sentence_idx' : i
                            ,'text' : text_n
                            ,'color' : 'red'
                            ,'embedder_score': text_links[i]['embedder_score']
                            ,'LLM_response' : llm_response})
        
    return answer

async def launch_app(update, context):
    if os.path.isfile("downloaded_file.pdf") and os.path.isfile("request_file.txt"):
        await update.message.reply_text('Приложение работает, пожалуйста подождите')
        request = open('request_file.txt', 'r').read()
        document = fitz.open("downloaded_file.pdf")

        text = prepare_text(document)

        sentences = answer(request, text) 

        modifi_document(sentences, document)
        await update.message.reply_text('Приложение закончило работу, можно качать файл')
    else:
        await update.message.reply_text('Вы еще не отправили один из файлов')


            

def main() -> None:

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    
    application.add_handler(MessageHandler(filters.Document.PDF, downloader))
    application.add_handler(CommandHandler('launch_app', launch_app))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("download", send_document))
    application.add_handler(CommandHandler("rerun", rerun))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    

    application.run_polling()

if __name__ == '__main__':
    main()

