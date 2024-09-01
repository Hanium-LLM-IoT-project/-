from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import logging
from logging.handlers import RotatingFileHandler
import re
import requests
from datetime import datetime, timedelta
import openai
import threading

# 기상관련
import requests
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import time

# Flask 앱 초기화
app = Flask(__name__)

# 로깅 설정
handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Firebase Admin SDK 초기화
cred = credentials.Certificate("firebasekey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hanium-d0dea-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# LLM 모델 생성
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o-mini", 
    openai_api_key="개인 정보 삭제")

# 대화 기록을 저장할 스토어
store = {}

# 대화 기록 가져오기
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 프롬프트 설정
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 스마트홈의 중앙관리를 맡고 있는 스마트홈 개인비서야. 사용자의 요청에 따라 필요한 경우 적절한 함수를 호출할 수 있어."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

# RunnableWithMessageHistory 생성
with_message_history = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Firebase에서 함수 목록 가져오기
def get_functions_from_firebase():
    try:
        ref = db.reference('functions')
        functions = ref.get()
        app.logger.info(f"Functions retrieved: {functions}")  # 로그 추가   
        return functions if functions is not None else {}
    except Exception as e:
        app.logger.error(f"Error retrieving functions: {e}")  # 오류 로그 추가
        return {}

def handle_user_message(session_id, user_input):
    functions = get_functions_from_firebase()
    app.logger.info(f"Functions available: {functions}")  # 함수 목록 로그 추가
    function_descriptions = "\n".join([f"{name}: {info.get('description', '')}" for name, info in functions.items()])

    # 프롬프트 작성: 함수 목록과 사용자 요청 포함
    prompt_with_functions = f"""
    다음은 사용할 수 있는 함수 목록입니다:
    {function_descriptions}
    사용자 요청: '{user_input}'

    너는 사용자와 일상적인 대화를 할 수 있어야 해. 대화의 맥락에 따라 적절한 반응을 하되, 사용자의 감정이나 상황에 맞는 응답을 제공해야 해.

    예시:
    - "집에 들어왔어" - "고생 많으셨어요! 오늘 하루는 어땠나요?"
    - "나 요즘 힘들어" - "힘든 시간을 보내고 계시군요. 괜찮으신가요? 언제든지 이야기해 주세요."


    일상적인 대화에서는 자연스럽게 상대방을 위로하거나 격려하는 내용을 포함해야 해. 
    그리고 사용자가 노래를 추천해달라고 할 때 추천해줄 수 있어야 해.
    사용자의 요청을 이해하고 분석한 후, 필요한 경우 사용할 수 있는 함수를 적절히 호출하도록 해. 요청에 대한 응답은 사람처럼 자연스럽게 작성해주고, 아래 형식으로 함수 이름과 상태(True/False)를 포함해줘:
    사용자가 집 밖으로 나간다고 할 때 거실 불을 꺼둘지 반드시 물어봐야해.

    예시:
    - "방의 불을 켜야 한다고 생각해요." - 함수 이름: [함수이름], 상태: [True]
    - "방의 불을 꺼야 할 것 같아요." - 함수 이름: [함수이름], 상태: [False]
    - "나 이제 나갈게" - 함수 이름: [함수이름], 상태: [False]
    - "나 이제 출근할게" - 함수 이름: [함수이름], 상태: [False]
    대화의 흐름을 유지하면서 필요한 경우 적절한 행동을 취할 수 있도록 답변을 작성해주세요.
    답변의 길이는 150자 이하로 되도록 해줘.
    사용자가 보낸 텍스트를 다시 말하면서 확인하는 건 하지 말아줘.

    """

    result = with_message_history.invoke(
        {"input": prompt_with_functions},
        {"configurable": {"session_id": session_id}}
    )
    response_message = result.content
    app.logger.info(f"LLM response: {response_message}")  # LLM 응답 로그 추가

    # LLM의 응답에서 함수 이름과 상태를 추출
    matched_function = None
    status = None

    # 응답에서 함수 이름과 상태를 찾기 위한 정규 표현식 사용
    function_match = re.search(r'함수 이름:\s*([^\s,]+)', response_message)
    status_match = re.search(r'상태:\s*(True|False)', response_message)

    # 매칭된 함수와 상태를 추출
    if function_match:
        matched_function = function_match.group(1).strip()  # 함수 이름 추출
    if status_match:
        status = status_match.group(1).lower().strip()  # 상태를 Boolean 값으로 변환

    app.logger.info(f"Matched function: {matched_function}, Status: {status}")  # 매칭된 함수 및 상태 로그 추가

    # 함수가 필요한 경우에만 실행
    if matched_function and matched_function in functions:
        # Firebase에서 해당 함수의 execution 상태 업데이트
        def update_function_execution(function_name, status):
            ref = db.reference(f'functions/{function_name}/execution')
            ref.set(status)

        # 함수 실행 상태 업데이트
        update_function_execution(matched_function, status == 'true')

        if status == 'true':
            return f"{matched_function} 함수가 켜졌습니다."
        else:
            return f"{matched_function} 함수가 꺼졌습니다."

    # 함수가 필요하지 않은 경우, 자연스러운 응답을 반환
    return response_message  # 자연스러운 응답 반환

######################################################################

def get_function_descriptions():
    functions_ref = db.reference('functions')
    functions_info = functions_ref.get()

    if functions_info is None:
        print("Error: functions 정보가 없습니다.")
        return []

    descriptions = []
    for function_name, function_data in functions_info.items():
        description = function_data.get('description', '설명이 없습니다.')
        descriptions.append(f"{function_name}: {description}")

    return "\n".join(descriptions)

def fetch_weather_from_firebase():
    try:
        weather_ref = db.reference('weather')
        return weather_ref.get() or {}
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {}

def is_window_open():
    try:
        window_ref = db.reference('functions/ServoAngle/execution')
        return window_ref.get() == True
    except Exception as e:
        print(f"Error checking window status: {e}")
        return False

def call_gpt(prompt):
    try:
        response = llm.invoke(prompt)  # prompt를 직접 전달
        content = response.content  # LLM의 응답에서 텍스트 추출
        print(f"GPT Response: {content}")  # 응답 로그 추가
        return content.replace(", ", "\n")  # 각 항목을 줄 바꿈으로 구분하여 반환
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return "함수 이름: update_window_status\n상태: [False]"  # 기본 응답 형식 수정

def fetch_weather_and_check_window():
    while True:
        weather_info = fetch_weather_from_firebase()
        precipitation_type = int(weather_info.get("precipitation_type", "0"))
        window_status = is_window_open()
        
        # 함수 설명 가져오기
        function_descriptions = get_function_descriptions()
        
        prompt = f"""
        너는 스마트홈 개인비서입니다. 현재 강수 유형은 {precipitation_type} 이 값이고, 창문 상태: {'열려있음' if window_status else '닫혀있음'}입니다.
        강수 유형이 1에서 4(비, 진눈깨비, 눈, 소나기)의 값을 가졌을 때만 창문을 닫아야 합니다.
        강수 유형이 '0'인 경우에는 창문을 닫지 말고 아무것도 반환하지 말아줘 
        
        다음 함수들이 있습니다:
        {function_descriptions} 

        이걸 참고해서 너가 호출해야 하는 함수를
        '- 함수 이름: 함수이름, 상태: False' 형태로 반환해야 해.
        이 과정을 필요할 때만 진행될거야.강수유형이 0일 때에는 아무것도 반환하지마.이게 제일 중요해.
        """
        
        gpt_response = call_gpt(prompt)
        print(gpt_response)

        matched_function, status = parse_gpt_response(gpt_response)

        if matched_function is not None and status is not None:
            update_function_execution(matched_function, status == 'True')

        time.sleep(50)

def parse_gpt_response(response):
    if not response:
        print("Error: GPT 응답이 비어 있습니다.")
        return None, None

    lines = response.splitlines()
    
    if len(lines) < 2:
        print("Warning: GPT 응답이 2줄 미만입니다. 한 줄로 처리합니다.")
        lines.append("상태: [False]")  # 기본 상태 추가

    try:
        function_name_line = lines[0]
        status_line = lines[1]

        if ":" not in function_name_line or ":" not in status_line:
            print("Error: GPT 응답 형식이 올바르지 않습니다. ':'가 누락되었습니다.")
            return None, None

        function_name = function_name_line.split(": ")[1].strip()  # 함수 이름 추출
        status = status_line.split(": ")[1].strip()  # 상태 추출
        
        return function_name, status
    except IndexError as e:
        print(f"Error: GPT 응답 형식이 올바르지 않습니다. 상세 오류: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


def update_function_execution(function_name, status):
    try:
        print(f"Updating function '{function_name}' to status '{status}'")
        ref = db.reference(f'functions/{function_name}/execution')
        ref.set(status)
        #print(f"Function '{function_name}' updated successfully.")
    except Exception as e:
        print(f"Error updating function '{function_name}': {e}")

# 프로그램 시작
#fetch_weather_and_check_window()

def start_background_thread():
    threading.Thread(target=fetch_weather_and_check_window, daemon=True).start()

###################################################################


def get_weather_forecast():
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'

    # 현재 날짜와 시간을 사용하여 base_date와 base_time을 설정
    base_date = datetime.now().strftime("%Y%m%d")
    base_time = (datetime.now() - timedelta(hours=1)).strftime("%H00")

    params = {
        'serviceKey': '개인정보 삭제',
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': '60',
        'ny': '127'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        try:
            print("Weather data received successfully.")
            json_data = response.json()

            weather_info = {
                'temperature': None,
                'precipitation_type': None,
                'humidity': None
            }

            for item in json_data['response']['body']['items']['item']:
                category = item['category']
                value = item['obsrValue']

                if category == 'T1H':  # 기온
                    weather_info['temperature'] = value
                elif category == 'PTY':  # 강수형태
                    weather_info['precipitation_type'] = value
                elif category == 'REH':  # 하늘상태
                    weather_info['humidity'] = value

            print("Parsed Weather Info:", weather_info)
            return weather_info
        except json.JSONDecodeError as e:
            print("Error parsing JSON response:", e)
    else:
        print(f"Failed to retrieve weather data. Status code: {response.status_code}")

    return None

# Firebase에 날씨 정보 업데이트
def update_weather_in_firebase():
    weather_info = get_weather_forecast()
    if weather_info:
        weather_ref = db.reference('weather')
        weather_ref.update({
            'temperature': weather_info['temperature'],
            'precipitation_type': weather_info['precipitation_type'],
            'humidity': weather_info['humidity']
        })
        app.logger.info(f"Weather updated in Firebase: {weather_info}")

# 사용자가 날씨 정보를 요청할 때 호출되는 엔드포인트
@app.route('/update_weather', methods=['GET'])
def update_weather():
    try:
        weather_info = get_weather_forecast()  # 날씨 정보를 가져옴
        if weather_info:
            update_weather_in_firebase()  # Firebase에 날씨 정보를 업데이트
            return jsonify({"message": "Weather information updated successfully", "weather": weather_info}), 200
        else:
            return jsonify({"error": "Failed to retrieve weather information"}), 500
    except Exception as e:
        app.logger.error(f"Error updating weather: {e}")
        return jsonify({"error": "Failed to update weather information"}), 500
# 일정 시간 간격으로 날씨 확인 및 업데이트
scheduler = BackgroundScheduler()
scheduler.add_job(update_weather_in_firebase, 'interval', hours=1)
scheduler.start()


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()

    user_input = data.get('text')
    session_id = data.get('session_id', 'default')  # 세션 ID 기본값을 설정

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    # 사용자 메시지 처리
    try:
        bot_response = handle_user_message(session_id, user_input)
        return jsonify({"responseText": bot_response})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400  # 오류 메시지 반환

if __name__ == '__main__':
    start_background_thread() 
    app.run(host='0.0.0.0', port=5000, debug=True)