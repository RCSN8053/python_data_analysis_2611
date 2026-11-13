#의료 기초 데이터 분석 미니 프로젝트
#numpy + pandas + matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# import matplotlib.font_manager as fm

#옵션
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 200)

#데이터
data = {
    '측정일': [
        '2026-02-01','2026-02-01','2026-02-02',
        '2026-02-02','2026-02-03','2026-02-03',
        '2026-02-04','2026-02-04','2026-02-05',
        '2026-02-05','2026-02-06','2026-02-06',
        '2026-02-07','2026-02-07','2026-02-08'
    ],
    '환자ID': [
        'P001','P002','P003','P004','P005',
        'P006','P007','P008','P009','P010',
        'P011','P012','P013','P014','P015'
    ],
    '성별': [
        'F','M','M','F','F','M','M','F',
        'F','M','F','M','M','F','F'
    ],
    '나이': [
        25,31,44,np.nan,52,39,28,33,
        47,61,29,42,np.nan,36,55
    ],
    '키': [
        '162cm','175cm','168cm','160cm','158cm',
        '180cm','172cm','165cm','170cm','177cm',
        '159cm','174cm','169cm','163cm','157cm'
    ],
    '몸무게': [
        '54kg','78kg','70kg','52kg','60kg',
        '90kg','68kg','58kg','73kg','82kg',
        '50kg','76kg','71kg','57kg','66kg'
    ],
    'SBP(수축기)': [
        108,135,142,118,150,130,125,112,
        140,160,105,138,145,120,155
    ],
    'DBP(이완기)': [
        70,88,92,75,95,85,80,72,
        90,100,68,89,96,78,97
    ],
    '심박수': [
        72,84,90,68,95,78,70,75,
        88,92,66,80,86,73,89
    ],
    '공복혈당': [
        92,110,125,98,130,np.nan,105,90,
        118,140,88,112,128,100,np.nan
    ],
    '흡연': [
        'N','Y','Y','N','N','Y','N','N',
        'Y','Y','N','Y','Y','N','N'
    ],
    '운동빈도(주)': [
        3,1,0,2,np.nan,1,4,2,
        1,0,5,1,0,3,np.nan
    ],
}

#데이터 탐색
df = pd.DataFrame(data)

print('[head()]')
print(df.head())

print('\ninfo')
print(df.info())

print('\n[describe] / 숫자 컬럼')
num_cols = ['나이', 'SBP(수축기)', 'DBP(이완기)', '심박수', '공복혈당', '운동빈도(주)']
print(df[num_cols].describe)



#데이터 정제 및 변환

#pd.to_datetime(): 문자열을 날짜/시간 타입으로 변환
df['측정일'] = pd.to_datetime(df['측정일'])
# print(df['측정일'].dtype)

#str.replace(): 문자열 치환
df['키_cm'] = df['키'].str.replace('cm', '', regex=False).astype(int)
df['몸무게_kg'] = df['몸무게'].str.replace('kg', '', regex=False).astype(int)
#문자열이라 분석에 분리하여 int형으로 변환

#결측치 확인
print(df.isnull().sum())
#결측치 처리시 데이터에 따라 방식을 다르게 함

df['나이'] = df['나이'].fillna(df['나이'].mean())
df['공복혈당'] = df['공복혈당'].fillna(df['공복혈당'].median())
df['운동빈도(주)'] = df['운동빈도(주)'].fillna(0)

#BMI추가(몸무게 / 키 ^2)
df['키_m'] = df['키_cm']/100
df['BMI'] = df['몸무게_kg'] / (df['키_m']**2)

# print(df[['키_cm', '키_m']].head())
# print(df['BMI'].head())

#맥박압 추가 수축기 - 이완기(혈과 탄성/압력 차이를 보는 간단 지표)
df['PP(맥박압)'] = df['SBP(수축기)'] - df['DBP(이완기)']
# print(df['PP(맥박압)'].head())

#평균동맥압(MAP) = (SBP + 2*DNP) / 3
df['MAP(평균동맥압)'] = (df['SBP(수축기)'] + 2* df['DBP(이완기)']) / 3
# print(df['MAP(평균동맥압)'].head())

'''
기준(간단버전)
- 정상 : SBP < 120 and DBP < 80
- 주위(상승) : SBP 120~129 and DBP < 80
- 고혈압 1기 : SBP 130~139 or DBP 80~89
- 고혈압 2기 : SBP >= 140 or DBP >= 90
'''

def bp_category(sbp, dbp):
    if sbp >= 140 or dbp >=90:
        return '고혈압 2기'
    elif 130 <= sbp <= 139 or 80 <= dbp <= 89:
        return '고혈압 1기'
    elif 120 <= sbp <= 129 and dbp < 80:
        return '주위(상승)'
    elif sbp < 120 and dbp < 80:
        return '정상'
    else:
        return '기타'

'''
def test(row):
    return 'test'

# lambda : 이름 없이 간단하게 작성하는 함수(익명 함수)
# lambda 매개변수 : 반환할 표현식
lambda row: 'test'

'''
#apply() : 행 단위로 함수를 적용할 때 자주 사용
df['혈압분류'] = df.apply(lambda row: bp_category(row['SBP(수축기)'], row['DBP(이완기)']), axis=1)

print('[혈압 분류 결과]')
print(df[['SBP(수축기)', 'DBP(이완기)', '혈압분류']].head())

'''
BMI 분류
- 저체중 : <18.5
- 정상 : 18.5 ~ 22.9
- 과체중 : 23.0 ~ 24.9
- 비만 : >= 25.0
'''

def bmi_category(bmi):
    if bmi < 18.5:
        return '저체중'
    elif bmi < 22.9:
        return '정상'
    elif bmi < 24.9:
        return '과체중'
    else:
        return '비만'

df['BMI분류'] = df['BMI'].apply(bmi_category)
print('[BMI 분류 결과]')
print(df[['키', '몸무게', 'BMI', 'BMI분류']].head())


#최고/최저
max_bmi_row = df.loc[df["BMI"].idxmax()] #최댓값 위치 반환
min_bmi_row = df.loc[df["BMI"].idxmin()] #최솟값 위치 반환

print('[BMI 최고]')
print(max_bmi_row[['환자ID','성별','나이','키','몸무게','BMI','BMI분류']])
print('[BMI 최저]')
print(min_bmi_row[['환자ID','성별','나이','키','몸무게','BMI','BMI분류']])

#혈압 분류별 평균 지표
#groupby(): 데이터를 같은 기준으로 묶는 기능
bp_group = df.groupby('혈압분류')[['SBP(수축기)','DBP(이완기)','PP(맥박압)','MAP(평균동맥압)','BMI','운동빈도(주)']].mean()
print("[혈압 분류별 평균]")
print(bp_group)

#성별 별 평균 지표
gender_mean = df.groupby('성별')[['SBP(수축기)','DBP(이완기)','PP(맥박압)','MAP(평균동맥압)','BMI','운동빈도(주)']].mean()
print("[성별 평균]")
print(gender_mean)

#나이대 평균
bins = [0,29,39,49,59,120]#경계값
labels = ['20대','30대','40대','50대','60대이상']
df['나이대'] = pd.cut(df['나이'], bins=bins, labels=labels, right=True)
age_group = df.groupby("나이대")[['SBP(수축기)', 'DBP(이완기)', '공복혈당', 'BMI']].mean()
print("나이대 별 평균")
print(age_group)

'''
corr(): 상관계수
1에 가까울 수록 같이 오르는 경향(양의 상관관계)
-1에 가까울 수록 반대로 움직이는 경향(음의 상관관계)
0에 가까우면 관계가 거의 없다.
'''

corr_cels = ['나이','키_cm','몸무게_kg','SBP(수축기)','DBP(이완기)','공복혈당','운동빈도(주)','BMI','PP(맥박압)','MAP(평균동맥압)']
corr_df = df[corr_cels].corr()
print('[상관관계]')
print(corr_df)

#조건 필터링
'''
예시 위험군 조건
- 위험 : 고혈압 2기
- 공복혈당 : 126 이상(당뇨기준)
- BMI : 25이상(비만)
조건 중 하나라도 True면 관심 대상으로 가정
'''
rist_df =df[
    (df['혈압분류'] == '고혈압 2기')|
    (df['공복혈당'] >= 126)|
    (df['BMI'] >= 25.0)].copy()
print('[관심대상 목록]')
print(rist_df[['환자ID','성별','나이','SBP(수축기)','DBP(이완기)','혈압분류','공복혈당','BMI','BMI분류','운동빈도(주)']])

#교차 분석 (성별 + 혈압분류)
'''
두개의 기준을 동시에 비교하는 분석
- 성별에 따라 횰압 분포가 어떻게 다른지 확인
- 단순 평균보다 더 분포를 볼 수 있음
'''
#crosstab() : 두개 이상의 범주형 변수의 관계를 표로 정리하는 함수
cross = pd.crosstab(df['성별'], df['혈압분류'])
print(cross)

#비율분석
'''
개수보다 비율이 중요할때가 많음
- 전체 중에서 각 혈압분류가 몇 %인지 등
'''
#value_counts() : 각 항목이 몇번 나왔는지 세는 함수
#normalize=True 추가시 개수가 아니라 비율로 바꿔줌
bp_ratio = df['혈압분류'].value_counts(normalize=True) *100
print("[혈압 분류 비율]")
print(bp_ratio.round(1))

#상위 N명 분석
'''
특정 기준에서 상위 데이터 확인
- BMI 높은 사람 상위 3명
- 위험군 빠륵 파악
'''
#nlargest() : 데이터에서 값이 가장 큰 상위 N개 뽑는 함수
top3_bmi = df.nlargest(3,'BMI')
print("[BMI 상위 3명]")
print(top3_bmi[['환자ID','BMI','BMI분류']])

#조건 + 그룹 분석
'''
특정 조건을 만족하는 데이터만 따로 분석
- 고혈압2기 환자만 따로 모와서 성별 평균 비교
'''
ht_df = df[df['혈압분류'] == '고혈압 2기']
ht_mean = ht_df.groupby('성별')[['BMI','공복혈당']].mean()
print('[고혈압2기 환자 - 성별 평균]')
print(ht_mean)

#순위 만들기
'''
데이터에 순서를 부여
- BMI 기준 순위
'''
#rank() : 순위를 매기는 함수
#ascending=True는 작은 값부터 1로 순위를 매김
df['BMI순위'] = df['BMI'].rank(ascending=True)
print("[BMI 순위]")
print(df[['환자ID','BMI','BMI순위']])

df_sorted = df.sort_values('BMI순위')
print(df_sorted[['환자ID', 'BMI', 'BMI순위']])

'''
조건을 점수화해서 위험도를 표현

조건
- 고혈합 2기
- 공복혈당 >=126
- BMI >= 25
- 각 조건마다 1점
'''

df['위험점수'] = (
    (df['혈압분류'] == '고혈압 2기').astype(int) +
    (df['공복혈당'] >= 126).astype(int) +
    (df['BMI'] >= 25).astype(int)
)

print('[위험점수]')
print(df[['환자ID', '위험점수']])

#시각화
#혈압 분포 히스토그램
'''
히스토그램
- 숫자 데이터가 어느 구간에 얼마나 많이 모여있는지 보여주는 그래프
- bin이 크면 클 수록 나뉘는 구간이 촘촘함
- 구간 수색에 따라 그래프 모양이 달라짐

'''
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
plt.hist(df['SBP(수축기)'],bins=8)
plt.title('SBP(수축기 혈압) 분포')
plt.xlabel('SBP(mmHg)')
plt.ylabel('빈도')
plt.tight_layout()
# plt.show()

#할압분류별 평균 막대 그래프
bp_sbp_mean = df.groupby('혈압분류')['SBP(수축기)'].mean().sort_values(ascending=True)
plt.figure()
plt.bar(bp_sbp_mean.index, bp_sbp_mean.values)
plt.title('AVG SBP By Blood Category')
plt.xlabel('Blood Pressure Category')
plt.ylabel('Average SBP(mmHg)')
plt.tight_layout()
# plt.show()

#산점도 BMI SBP
'''
산점도 : 두개의 숫자 변수 관계를 점으로 표현한 그래프
'''
plt.figure()
plt.scatter(df['BMI'],df['SBP(수축기)'])
plt.title("BMI VS SBP scatter plot")
plt.xlabel("BMI")
plt.ylabel("SBP")
plt.tight_layout()
# plt.show()

#측정일 기준 SBP 추이
date_sbp = df.groupby('측정일')['SBP(수축기)'].mean()
plt.close('all')
plt.figure()
plt.plot(date_sbp.index, date_sbp.values)
plt.title('AVG SBP Over Time')
plt.xlabel('date')
plt.xticks(rotation=30)
plt.ylabel('AVG SBP')
plt.tight_layout()
# plt.show()

#BMI 분류 인원수(pie)
bmi_count = df['BMI분류'].value_counts()
plt.figure
plt.pie(bmi_count.values,labels=bmi_count.index,autopct='%1.1f%%')
plt.title("BMI Category DIST")
plt.tight_layout()
# plt.show()

#혈압 분류별 BMI
'''
박스 플롯
- 평균이 아니라 분포 보여줌
- 이상치 확인 가능
'''
plt.close('all')
plt.figure()
df.boxplot(column='BMI', by='혈압분류')
plt.title('BMI DIST by Blood category')
plt.suptitle('')
plt.xlabel('Boold category')
plt.ylabel('BMI')
plt.tight_layout()
# plt.show()

#성별 BMI
plt.close('all')
plt.figure()
df[df['성별'] == 'M']['BMI'].hist(alpha=0.5,bins=8)
df[df['성별'] == 'F']['BMI'].hist(alpha=0.5,bins=8)
plt.title("BMI DIST By Gender")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.legend(['Mael','Female'])
plt.tight_layout()
# plt.show()

#상관관계 히트맵
'''
히트맵
- 값의 크기를 색으로 표현하는 그래프
- 값이 클수록 진한 색
- 값이 작을수록 연한 색
'''
plt.close('all')
corr_cols = ['나이','키_cm','몸무게_kg','SBP(수축기)','DBP(이완기)','공복혈당','BMI']
corr = df[corr_cols].corr()
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.yticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.title('correlation Heatmap')
plt.tight_layout()
# plt.show()

#인사이트 자동 요약

#혈압분류 비율
bp_counts = df['혈압분류'].value_counts()
bp_ratio = (bp_counts / len(df))*100
print("[혈압 분류 분포(개수)]")
print(bp_counts)
print("[혈압 분류 분포(%)]")
print(bp_ratio.round(1))

#위험 신호 : 공복혈당 평균과 126이상 비율
mean_glu = df['공복혈당'].mean()
high_glu_ratio = (df['공복혈당'] >= 126).mean() *100
print(f'[공복혈당 평균 : {mean_glu: .2f}mg/dl]')
print(f'[공복혈당 126이상 비율 : {high_glu_ratio: .1f}%]')

#운동빈도와 주요 지표 관계
corr_exercise_sbp = df['운동빈도(주)'].corr(df['SBP(수축기)'])
corr_exercise_bmi = df['운동빈도(주)'].corr(df['BMI'])
print(f"[운동빈도 / SBP 상관계수] : {corr_exercise_sbp: .2f}")
print(f"[운동빈도 / BMI 상관계수] : {corr_exercise_bmi: .2f}")

#전략/제안

stratege = []

#혈압 2기 비율 기반 코멘트
ht2_ratio = (df['혈압분류'] == '고혈압 2기').mean()*100
if ht2_ratio >= 30:
    stratege.append('고혈압 2기 비율이 높은 편이라, 생활습관관련 교육 및 추적 관리가 필요함')
else:
    stratege.append('고혈압 2기 비율이 과도하가 높진 않지만, 고혈압 1기 이상에 대한 예방 안내가 유효합니다.')

#BMI 비만 비율 코멘트
obesity_ratio = (df['BMI분류'] == '비만').mean()*100
stratege.append(f'BMI 기준 비만 비율이 {obesity_ratio: .1f}%로 확인되어 체중관리 프로램을 고려할 수 있습니다.')

#혈당 고위험 비율 코멘트
if high_glu_ratio > 20:
    stratege.append(f'공복혈당 126 이상 비율이 {high_glu_ratio: 1f}이기 때문에 혈당 관려 추가 검사/상담 연계를 고려할 수 있습니다.')
else:
    stratege.append('공복할당 고위험 비율이 높진 않지만, 식습관, 활동량 관련 기본 가이드를 제공하면 좋습니다.')

print('\n[3가지 제한]')
for s in stratege:
    print(f'- {s}')

#위험 점수 기반 위험군 요약
'''
- 위험 점수 2 이상 == 고위험군
'''

high_risk = df[df['위험점수'] >= 2]
high_risk_ratio = (len(high_risk) / len(df))*100

print(f'[고위험군 비율(위험점수 2 이상)] : {high_risk_ratio: .1f}')
print('[고위험군 목록]')
print(high_risk[['환자ID', 'BMI', '공복혈당', '혈압분류', '위험점수']])

#최상위 위험 대상 TOP 3
'''
가장 위험한 사람 추출
- 위험점수 기준
- 돌일하면 BMI 높은순
'''
top_risk = df.sort_values(['위험점수', 'BMI'], ascending=False).head(3)
print('[최상위 위험 대상 Top 3]')
print(top_risk[['환자ID','위험점수','BMI','공복혈당','혈압분류']])

#전체 건강 상태 등급 평가
'''
데이터 전체를 하나의 결과로 요약
기준
- 고위험군 비율
- 혈당 위험 비율
'''
if high_risk_ratio > 30 or high_glu_ratio > 30:
    health_level = '위험'
elif high_risk_ratio > 15 or high_glu_ratio > 15:
    health_level = '주의'
else:
    health_level = '양호'

print(f'[전체 건강 상태 평가] : {health_level}')