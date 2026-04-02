import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# 페이지 기본 설정
st.set_page_config(page_title="자전거 수요 예측", layout="wide")

st.title("자전거 수요 예측 (GRU Model)")
st.write("날짜와 기상 정보를 입력하면 하루(24시간)의 자전거 대여량 흐름을 예측합니다.")

# 1. 모델 및 스케일러 로드
@st.cache_resource  # 캐싱을 사용하여 리소스 로딩 속도 최적화
def load_assets():
    model_path = './model/bike_gru_model.h5'
    scaler_x_path = './model/scaler_X.pkl'
    scaler_y_path = './model/scaler_y.pkl'
    
    # 파일 존재 여부 확인
    if not os.path.exists(model_path) or not os.path.exists(scaler_x_path):
        return None, None, None

    # 모델 및 스케일러 로드 (compile=False 필수)
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()

if model is None:
    st.error("모델 파일이 없습니다. 먼저 'train.py(가정)'를 실행하여 모델과 스케일러를 생성해주세요.")
else:
    # 2. 사용자 입력 받기 (Sidebar)
    st.sidebar.header("조건 입력")
    
    # 날짜
    date_input = st.sidebar.date_input("날짜 선택")
    
    # 근무일/공휴일
    day_type = st.sidebar.radio(
        "오늘의 유형은 무엇인가요?",
        ["평일 (근무일)", "주말 (휴일 아님)", "공휴일"]
    )
    
    # 선택에 따른 holiday, workingday 변수 자동 할당
    if day_type == "평일 (근무일)":
        workingday = 1
        holiday = 0
    elif day_type == "주말 (휴일 아님)":
        workingday = 0
        holiday = 0
    else: # 공휴일
        workingday = 0
        holiday = 1
    
    # 계절
    season = st.sidebar.selectbox("계절", [1, 2, 3, 4], format_func=lambda x: {1:'봄', 2:'여름', 3:'가을', 4:'겨울'}[x])
    
    # 날씨
    weather = st.sidebar.selectbox("날씨", [1, 2, 3, 4], format_func=lambda x: {1:'맑음', 2:'흐림/안개', 3:'가벼운 눈/비', 4:'폭우/폭설'}[x])
    
    # 기상 데이터
    # (참고: 실제 서비스에서는 시간대별로 기온이 다르겠지만, 시뮬레이션 편의상 하루 평균값으로 가정합니다)
    temp = st.sidebar.slider("기온 (℃)", -10.0, 40.0, 20.0)
    atemp = st.sidebar.slider("체감 기온 (℃)", -10.0, 45.0, 22.0)
    humidity = st.sidebar.slider("습도 (%)", 0, 100, 50)
    windspeed = st.sidebar.slider("풍속", 0.0, 60.0, 10.0)

    # 3. 예측 실행
    if st.button("하루 전체 예측 보기"):
        
        # 0시부터 23시까지 반복 예측을 위한 리스트 생성
        predicted_results = []
        hours_range = range(24)
        
        # 진행률 표시줄 (선택사항)
        progress_bar = st.progress(0)
        
        for target_hour in hours_range:
            # 날짜 정보 처리
            datetime_str = f"{date_input} {target_hour:02d}:00"
            datetime_obj = pd.to_datetime(datetime_str)
            
            day = datetime_obj.day
            month = datetime_obj.month
            
            # 중요: 연도를 2012년으로 고정
            # 이유: 학습 데이터가 2011~2012년뿐이라 2025년을 넣으면 스케일러가 값을 15배로 인식해 결과가 폭증함
            year = 2012 
            
            dayofweek = datetime_obj.dayofweek
            
            # 기본 입력 벡터 생성 (target_hour 기준)
            input_features = np.array([[
                season, 
                int(holiday), 
                int(workingday), 
                weather, 
                temp, 
                atemp, 
                humidity, 
                windspeed,
                target_hour, # 현재 타겟 시간
                day,
                month,
                year,        # 고정된 연도 사용
                dayofweek
            ]])
            
            # 시퀀스 생성 로직 (중요: 과거 24시간의 시간 흐름을 만들어줌)
            TIME_STEPS = 24
            
            # 01. 타겟 시간의 데이터를 24번 복제
            input_seq_batch = np.repeat(input_features, TIME_STEPS, axis=0)
            
            # 02. 'Hour' 변수(인덱스 8)를 과거 시간으로 순차 변경
            # (예: 타겟이 02시라면 -> 전날 03시...오늘 01시, 02시 순서로 흐름 생성)
            hour_col_idx = 8
            for i in range(TIME_STEPS):
                past_hour = (target_hour - (TIME_STEPS - 1) + i) % 24
                input_seq_batch[i, hour_col_idx] = past_hour
            
            # 스케일링 (수정된 배치 데이터로 수행)
            try:
                input_scaled = scaler_X.transform(input_seq_batch)
            except ValueError as e:
                st.error(f"입력 데이터 형태 오류: {e}")
                st.stop()

            # 시계열 형태 변환 (1, 24, 13)
            model_input = input_scaled.reshape(1, TIME_STEPS, input_features.shape[1])
            
            # 예측 수행
            prediction = model.predict(model_input, verbose=0)
            
            # 역변환
            predicted_count = scaler_y.inverse_transform(prediction)
            
            # 결과 저장 (음수 제거)
            result_val = max(0, int(predicted_count[0][0]))
            predicted_results.append(result_val)
            
            # 진행률 업데이트
            progress_bar.progress((target_hour + 1) / 24)
            
        progress_bar.empty() # 진행률 바 제거

        # 결과 시각화 섹션
        st.divider() # 구분선
        
        # 1. 데이터프레임 생성
        df_result = pd.DataFrame({
            "시간 (Hour)": [f"{h:02d}:00" for h in hours_range],
            "예측 대여량": predicted_results
        })
        
        # 컬럼 2개로 나누어 그래프와 표 배치
        col1, col2 = st.columns([2, 1]) # 그래프를 더 넓게
        
        with col1:
            st.subheader("시간대별 대여량 그래프")
            # 스트림릿 내장 라인 차트 (인덱스를 시간으로 설정)
            st.line_chart(df_result.set_index("시간 (Hour)"))
            
            # 피크 타임 분석
            max_val = max(predicted_results)
            max_idx = predicted_results.index(max_val)
            peak_time = df_result.iloc[max_idx]["시간 (Hour)"]
            
            st.success(f"분석 결과: 가장 붐비는 시간은 **{peak_time}** 이며, 약 **{max_val}대**가 필요할 것으로 예상됩니다.")

        with col2:
            st.subheader("상세 예측 테이블")
            # 데이터프레임 표시 (높이 조절)
            st.dataframe(df_result, height=400, hide_index=True)
        
        # 입력 정보 확인용 (선택사항)
        st.info(f"기준 날짜: {date_input} (모델 적용 연도: 2012) / 유형: {day_type}")