from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import pandas as pd
from konlpy.tag import Okt
import csv
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
import random
import calendar
import pandas as pd
import glob

okt = Okt()

# 키워드 추출 함수 생성
def get_keyword(data, threshold = 0.5):
    
    documents = [' '.join(okt.nouns(title)) for title in data]


    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 단어와 해당 단어의 TF-IDF 값을 저장하는 딕셔너리 생성
    feature_names = vectorizer.get_feature_names_out()
    word_scores = defaultdict(float)

    for i in range(len(data)):
        feature_index = tfidf_matrix[i, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        for word_index, score in tfidf_scores:
            word = feature_names[word_index]
            word_scores[word] += score

    # 최대 TF-IDF 점수 계산
    max_score = max(word_scores.values())

    # 높은 TF-IDF 값을 가진 단어 3개를 출력
    sorted_words = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)
    top_words = sorted_words[:]

    keyword = []

    # print("가장 중요한 단어 10개:")
    for i, (word, score) in enumerate(top_words, 1):
        normalized_score = score / max_score  # 정규화
        normalized_score = round(normalized_score, 2)  # 소수점 2번째 자리까지 반올림
        keyword.append((word, int(normalized_score*100)))
        #if normalized_score >= threshold:
            #keyword.append((word, int(normalized_score*100)))
            # print(f'{i}. {word}: {int(normalized_score*100)}%')
    
    return keyword

# 뉴스 크롤링 함수 생성
def get_new_num(keywords, year, month):
    driver = webdriver.Chrome()
    result_list = []
    
    url = 'https://www.google.com/search?q={word}&rlz=1C5CHFA_enKR1040KR1040&biw=1512&bih=865&sxsrf=APwXEddHhcqCVIs4Xj8DdCgJyXSitpj6sg%3A1685686591319&source=lnt&tbs=cdr%3A1%2Ccd_min%3A{start_month}%2F{start_day}%2F{year}%2Ccd_max%3A{start_month}%2F{end_day}%2F{year}&tbm=nws'

    # timesleep = random.randint(1,10) 사용 안해서 주석처리

    # CSV 파일 열기
    # with open('result.csv', 'w', newline='') as file:
        # writer = csv.writer(file)
        # 컬럼명 쓰기
        # writer.writerow(["Keyword", "Year", "Month", "Number of articles"])
        
    for word, _ in keywords:
        _, end_day = calendar.monthrange(year, month) # 해당월의 마지막 날짜를 구하기
        new_url = url.format(word = word, start_month = month, start_day = 1, year = year, end_month = month, end_day = end_day)
        driver.get(new_url)
        time.sleep(4)
        driver.find_element(By.ID, 'hdtb-tls').click() # 도구를 통해 날짜를 조절하면 기사수를 볼 수 없었음 -> 때문에 다시 도구버튼 클릭해서 기사 수를 보게 함.
        time.sleep(4)
        results = driver.find_elements(By.CLASS_NAME, 'LHJvCe')
        
        for result in results:
            result_text = result.text # 기사 수 
            # 결과에서 숫자 부분만 추출
            num_articles = re.search(r'\d+', result_text.replace(',', '')).group()
            # writer.writerow([word, year, month, num_articles]) 
            result_list.append(num_articles) 
                
        # print(result_list)
    return result_list


# 메인 함수
# 데이터 확보 ==========================================================================================================

# 현재 디렉토리에 모든 csv파일의 경로 가져오기

lib_file_paths = glob.glob("./lib_data/*.xlsx")
book_file_paths = glob.glob("./book_data/*.xlsx")

lib_file_paths += glob.glob("./lib_data/*.csv")
book_file_paths += glob.glob("./book_data/*.csv")

#lib_file_paths = glob.glob('lib_data/**', recursive=True)
#book_file_paths = glob.glob('book_data/**', recursive=True)

sorted_lib_file_paths = sorted(lib_file_paths)
sorted_book_file_paths = sorted(book_file_paths)

print(sorted_lib_file_paths)
print(sorted_book_file_paths)

# CSV 파일들을 저장할 리스트 생성
lib_data_frames = []
book_data_frames = []

# 데이터를 통합해서 추출할 데이터 프레임
result_l_df = pd.DataFrame(columns=['키워드', '점수', '연도', '월', '데이터 출처'])
result_b_df = pd.DataFrame(columns=['키워드', '점수', '연도', '월', '데이터 출처'])

# 도서관 데이터 활용 (데이터 전처리 -> 키워드 추출 -> 관련 뉴스 수 확인 -> 데이터 저장)====================================================
month = 1
year = 2020
for l_file in sorted_lib_file_paths:
    if l_file.endswith('.csv'):
        df = pd.read_csv(l_file, encoding='cp949')
    elif (l_file.endswith('.xlsx') | l_file.endswith('.xls')):
        df = pd.read_excel(l_file)
    # '서명' 컬럼과 'KDC' 컬럼만 선택하여 데이터프레임에 추가
    df = df[['서명', 'KDC']]
    # KDC 컬럼의 값이 320에서 330 사이인 데이터만 선택
    df = df[(df['KDC'] >= 320) & (df['KDC'] <= 330)]
    # 조건에 해당하는 '서명'을 리스트로 추가
    threshold = 0.5
    keyword = get_keyword(df['서명'].tolist(), threshold)
    print(keyword)
    #new_num = get_new_num(keyword, year, month)

    # 데이터 출처 설정
    data_source = '도서관'

    # 새로운 데이터프레임에 행 추가
    i = 0
    for key, score in keyword:
        #new_row = {'키워드': key, '점수':score, '연도': year, '월': month, '기사 수': new_num[i], '데이터 출처': data_source}
        new_row = {'키워드': key, '점수': score,'연도': year, '월': month,  '데이터 출처': data_source}
        result_l_df = result_l_df.append(new_row, ignore_index=True)
        i += 1

    # 연도와 월 수정
    if (month == 12):
        month = 1
        year += 1
    else:
        month += 1
    if (year == 2023):
        break
    
result_l_df.to_csv('tf_lib_data.csv', index=False)


# 서점 데이터 활용 (데이터 전처리 -> 키워드 추출 -> 관련 뉴스 수 확인 -> 데이터 저장) ====================================================
month = 1
year = 2020
for b_file in sorted_book_file_paths:
    print(year, month)
    if b_file.endswith('.csv'):
        if(year == 2022):
            df = pd.read_csv(b_file, encoding='cp949')
        else:
            df = pd.read_csv(b_file)
    elif b_file.endswith('.xlsx'):
        df = pd.read_excel(b_file)

    # '상품명' 컬럼과 '부가기호' 컬럼만 선택하여 데이터프레임에 추가
    df = df[['상품명', '부가기호']]

    if(year == 2020):
        df['부가기호'].fillna('0',inplace=True)
        df['부가기호'] = df['부가기호'].replace('\r\n', '0')
        df['부가기호'] = df['부가기호'].astype(int)
        df['부가기호'] = df['부가기호'].astype(str)
        df['부가기호']=df['부가기호'].apply(lambda x: x if len(x) <= 5 & len(x) >=3 else '0')
        df['부가기호']=df['부가기호'].str.replace('\t', '0')
        df['부가기호'] = df['부가기호'].astype(int)
        df['부가기호'] = df['부가기호'].astype(str).str[-3:]
        df['부가기호'] = df['부가기호'].astype(str).str.strip()
        df = df[(df['부가기호'].astype(int) >= 320) & (df['부가기호'].astype(int) <= 330)]

    elif(year == 2021):
        df['부가기호'].fillna(0.0,inplace=True)
        df = df[pd.to_numeric(df['부가기호'], errors='coerce').notnull()]
        df['부가기호'] = df['부가기호'].astype(float)
        df['부가기호'] = df['부가기호'].astype(int)
        df['부가기호'] = df['부가기호'].astype(str)
        df['부가기호']=df['부가기호'].apply(lambda x: x if len(x) <= 5 & len(x) >=3 else '0')
        df['부가기호']=df['부가기호'].str.replace('\t', '0')
        df['부가기호']=df['부가기호'].str[-3:]
        df['부가기호'] = df['부가기호'].astype(str).str.strip()
        df['상품명'] = df['상품명'].astype(str)
        df = df[(df['부가기호'].astype(int) >= 320) & (df['부가기호'].astype(int) <= 330)]
        
    elif(year == 2022):
            # 부가기호 to KDC
        df['부가기호'].fillna('0',inplace=True)
        df = df[pd.to_numeric(df['부가기호'], errors='coerce').notnull()]
        df['부가기호'] = df['부가기호'].astype(float)
        df['부가기호'] = df['부가기호'].astype(int)
        df['부가기호'] = df['부가기호'].astype(str)
        df['부가기호']=df['부가기호'].apply(lambda x: x if len(x) <= 5 & len(x) >=3 else '0')
        df['부가기호']=df['부가기호'].str.replace('\t', '0')
        df['부가기호'] = df['부가기호'].str[-3:]
        df['부가기호'] = df['부가기호'].astype(str).str.strip()
        df = df[(df['부가기호'].astype(int) >= 320) & (df['부가기호'].astype(int) <= 330)]

    # 조건에 해당하는 '서명'을 리스트로 추가
    threshold = 0.5
    keyword = get_keyword(df['상품명'].tolist(), threshold)
    print(keyword)
    #new_num = get_new_num(keyword, year, month)

    # 데이터 출처 설정
    data_source = '서점'

    # 새로운 데이터프레임에 행 추가
    i = 0
    for key, score in keyword:
        #new_row = {'키워드': key, '점수':score, '연도': year, '월': month, '기사 수': new_num[i], '데이터 출처': data_source}
        new_row = {'키워드': key, '점수':score, '연도': year, '월': month, '데이터 출처': data_source}
        result_b_df = result_b_df.append(new_row, ignore_index=True)
        i += 1

    # 연도와 월 수정
    if (month == 12):
        month = 1
        result_b_df.to_csv(f'tf_book_data_{year}.csv', index=False)
        # 중간에 에러가 발생할 수 있어서 연도별로 저장하고 추후 병합하여 book_data.csv파일 만들어냄
        year += 1
        #result_b_df.to_csv(f'book_data_{year}.csv', index=False)
    else:
        month += 1
    if (year == 2023):
        break

result_b_df.to_csv(f'tf_book_data.csv', index=False)
# CSV 파일로 저장
#result_b_df.to_csv('book_data.csv', index=False)

# 시각화 =============================================================================================================
#dt = pd.read_csv('data_csv')



