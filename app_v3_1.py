import streamlit as st
import re
from sentence_transformers import SentenceTransformer, util
import requests
import uuid
import json

# tresh_hold = 0.78
auth = 'Mzc3ODA5ODEtYjMxZC00N2MxLTliODktODEzM2ExYWFlNDA4OjkzMmQzZjE5LWQzYzktNGQ1YS04Y2NlLTZlZGE5OThhYzgzMQ=='
model = SentenceTransformer('intfloat/multilingual-e5-large')
# model = SentenceTransformer('ai-forever/sbert_large_nlu_ru')

def prepare_links(link_text):
    new_txt_list = []

    text_2 = re.split(r'(?=[.]\s[А-ЯA-Z])', link_text)
    for i in text_2:
        new_txt_list.append(re.sub(r'([.]\s)', '', i))

    return new_txt_list


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


def main(text_query, links, tresh_hold):

    text_links_prep = prepare_links(links)

    text_links, list_cand = embeding(text_query, text_links_prep, tresh_hold)

    response = get_token(auth)
    if response != -1:
        giga_token = response.json()['access_token']

    answer = []

    for i in text_links:
        text_n = text_links[i]['text']

        if i in list_cand:
            # text_for_api = f'Ответь да или нет, есть ли подтверждение текста А в тексте Б\n\nТекст А это - {text_query}\n\nТекст Б это - {text_n}'
            text_for_api = f'Подтверждается ли текст "{text_query}" текстом "{text_n}"\nОтветь только "да" или "нет"'
            print(text_for_api)
            answer_n = get_chat_completion(giga_token, text_for_api)
            print(str(answer_n.json()['choices'][0]['message']['content']))
            answer.append({'sentence_idx' : i
                           ,'text' : text_n
                           ,'embedder_score': text_links[i]['embedder_score']
                           ,'LLM_response' : str(answer_n.json()['choices'][0]['message']['content'])}) 
            # print(answer_n.json()['choices'][0]['message']['content'], text_n)
        else: 
            answer.append({'sentence_idx' : i
                           ,'text' : text_n
                           ,'embedder_score': text_links[i]['embedder_score']
                           ,'LLM_response' : 'None'})

    
    for i in range(len(text_links_prep)):
        st.write(f'({i}) --> {text_links_prep[i]}')



    return answer, text_links_prep

st.title('App for validation links')
query = st.text_area('Text from article')
links = st.text_area('Text from link')  
tresh_hold = st.number_input('Enter your tresh_hold')

if st.button('Tap to submit'):

    st.title('Все предложения из источника')

    answer, text_links = main(query, links, tresh_hold)

    st.title('Ответы LLM')
    
    st.write(answer)
    # for i in answer:
    #     st.write(f'({i}) --> предложение-кандитат {text_links[i]}')
    #     if answer[i]['sentence_idx'] == 'Да':
    #         st.success(f'--> ответ модели: {answer[i]}')
    #     else:
    #         st.error(f'--> ответ модели: {answer[i]}')

