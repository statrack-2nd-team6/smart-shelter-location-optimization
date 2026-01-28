# pip install dvc

# # 1. DVC 초기화
# dvc init

# # 2. 이미지 폴더 추가 (1만 장의 이미지를 DVC가 관리하도록 설정)
# dvc add bus_stop_images

# # 3. Git에 변경사항 기록 (이미지 대신 메타데이터만 기록함)
# git add bus_stop_images.dvc .gitignore
# git commit -m "Add dataset using DVC"

# 1. 구글 드라이브 연동 라이브러리 설치 (이미 하셨다면 생략 가능)
pip install dvc-gdrive

# 2. 원격 저장소 추가 (ID 부분에 위에서 복사한 값을 넣으세요)
dvc remote add -d myremote gdrive://1_mVpz4ngt3H8GrGzbCgp3WI-A6J8AzF7

# 3. 설정 변경사항을 Git에 기록
git add .dvc/config
git commit -m "Configure Google Drive as DVC remote"