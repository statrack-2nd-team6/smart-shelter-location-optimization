import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# --- 1. 설정 및 경로 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "final_pu_dataset.csv")
IMG_DIR = os.path.join(BASE_DIR, "..", "bus_stop_images")

# --- 2. 데이터셋 클래스 ---
class ShelterDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), file_name

# --- 3. 전처리 및 로더 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ShelterDataset(CSV_PATH, IMG_DIR, transform=transform)
# CPU 환경을 고려해 batch_size를 조절하세요.
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --- 4. 모델 설정 (ResNet18) ---
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model = model.to("cpu")

# --- 5. 학습 ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 스마트 쉼터 분석 모델 학습 시작 (1 Epoch 테스트)...")
model.train()
for epoch in range(1):
    for i, (images, labels, _) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

# --- 6. 결과 추출 (Inference) ---
print("🔍 전체 후보지 점수 매기는 중...")
model.eval()
results = []
with torch.no_grad():
    for images, labels, filenames in dataloader:
        outputs = model(images).squeeze()
        # 점수가 리스트가 아닐 경우(배치 1일 때) 처리
        if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
        
        for name, score, label in zip(filenames, outputs, labels):
            if label == 0: # 아직 설치 안 된 곳(U)만 대상
                results.append({'file_name': name, 'score': score.item()})

# --- 7. 상위 10개 출력 및 저장 ---
top_10 = pd.DataFrame(results).sort_values(by='score', ascending=False).head(10)
print("\n🏆 [스마트 쉼터 설치 권장 TOP 10]")
print(top_10)

top_10.to_csv(os.path.join(BASE_DIR, "top_candidates.csv"), index=False, encoding='utf-8-sig')
print(f"\n✅ 분석 완료! 후보 리스트가 'top_candidates.csv'로 저장되었습니다.")