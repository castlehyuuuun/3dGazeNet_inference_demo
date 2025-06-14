# from pathlib import Path
# import numpy as np
#
# # 경로 설정
# DATA_ROOT = Path("C:/Users/user/Desktop/skh/ETRI")
# GZ_PATH = DATA_ROOT / "processed_videos"
# INFO_PATH = DATA_ROOT / "PNU_selected"
# test_subjects = {
#     "A2-1022-5", "A2-1026-3", "A2-1026-04", "A2-1029-02", "A2-1029-03", "A2-1029-04", "A2-1029-05"
# }
# # event_info에서 마지막 줄의 숫자 라벨 파싱
# def parse_event_label(event_info_path):
#     try:
#         with open(event_info_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             for line in reversed(lines):
#                 line = line.strip()
#                 if not line or line.startswith("#"):
#                     continue
#                 clean = line.split("#")[0].strip()
#                 if clean.isdigit():
#                     return int(clean)
#     except Exception as e:
#         print(f" label 로딩 실패 ({event_info_path}): {e}")
#     return None
#
# X_train, y_train, id_train = [], [], []
# X_test, y_test, id_test = [], [], []
#
# # 모든 event_info.txt를 재귀적으로 탐색
# for event_info_path in INFO_PATH.rglob("event_info.txt"):
#     try:
#         event_id = event_info_path.parent.name  # 예: 000, 001
#         label = parse_event_label(event_info_path)
#         if label is None:
#             continue
#
#         # 경로로부터 subject_name, session_id 추출
#         parts = event_info_path.parts
#         subject_name = parts[parts.index("PNU_selected") + 1]
#         session_id = parts[parts.index(subject_name) + 1]
#
#         print(f"\n subject: {subject_name}, session: {session_id}, event_id: {event_id} → label = {label}")
#
#         # processed_videos 폴더에서 일치하는 모든 폴더 탐색
#         expected_prefix = f"processed_{subject_name}_{session_id}_rec_{event_id}_"
#         matched_folders = [
#             f for f in GZ_PATH.iterdir()
#             if f.name.startswith(expected_prefix) and f.name.endswith("_out_resnet_x128_vertex")
#         ]
#
#         if not matched_folders:
#             print(f" No matched folder for {expected_prefix}")
#             continue
#
#         for folder in sorted(matched_folders):
#             print(f" 후보: {folder.name}")
#             gaze_txt_path = folder / "predicted_gaze_vectors.txt"
#             print(f" 찾은 경로: {gaze_txt_path}")
#
#             if gaze_txt_path.exists():
#                 try:
#                     gaze_seq = np.loadtxt(gaze_txt_path, delimiter=',')
#                     print(f" gaze shape: {gaze_seq.shape}")
#                     MAX_LEN = 300  # 전체 프레임 길이 통일
#
#                     if gaze_seq.shape[0] <= MAX_LEN:
#                         padded = np.zeros((MAX_LEN, 3), dtype=np.float32)
#                         padded[:gaze_seq.shape[0]] = gaze_seq  # 앞쪽에 실제 값 넣고 나머진 0
#                         X_data.append(padded)
#                         y_data.append(label)
#                         id_data.append(folder.name)
#                         print(" 저장 완료 (패딩)")
#                     else:
#                         print(f"⚠ 프레임 수 초과: {gaze_seq.shape[0]}")
#
#                 except Exception as e:
#                     print(f" gaze vector 로딩 실패: {e}")
#             else:
#                 print(" gaze 텍스트 파일 없음")
#
#     except Exception as e:
#         print(f" 예외 발생: {e}")
#
# print(f"\n 총 수집된 샘플 수: {len(X_data)}")
#
# ###########################
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
#
# #  CrossEntropyLoss가 처리 가능한 라벨만 필터링
# X_data_filtered, y_data_filtered, id_data_filtered = [], [], []
#
# for x, y, vid in zip(X_data, y_data, id_data):
#     if y in [0, 1]:
#         X_data_filtered.append(x)
#         y_data_filtered.append(y)
#         id_data_filtered.append(vid)
#
# target_list = X_test if subject_name in test_subjects else X_train
# label_list = y_test if subject_name in test_subjects else y_train
# id_list    = id_test if subject_name in test_subjects else id_train
#
# target_list.append(padded)
# label_list.append(label)
# id_list.append(folder.name)
#
#
#
# print(f" 최종 샘플 수: {len(X_data)} / 라벨 종류: {set(y_data)}")
#
# # -----------------------------
# #  Dataset 구성
# # -----------------------------
# class GazeDataset(Dataset):
#     def __init__(self, X, y, ids):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#         self.ids = ids  # 문자열 리스트
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx], self.ids[idx]
#
# # train_loader: 학습용 / eval_loader: 순서 고정된 평가용
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# id_train = np.array(id_train)
#
# X_test = np.array(X_test)
# y_test = np.array(y_test)
# id_test = np.array(id_test)
#
# train_dataset = GazeDataset(X_train, y_train, id_train)
# test_dataset = GazeDataset(X_test, y_test, id_test)
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
#
#
# # -----------------------------
# #  LSTM 모델 정의
# # -----------------------------
# class GazeLSTM(nn.Module):
#     def __init__(self, input_dim=3, hidden_dim=64, num_layers=1, num_classes=2):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])
#
# # -----------------------------
# #  학습 루프
# # -----------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GazeLSTM().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
# num_epochs = 100
# for epoch in range(num_epochs):
#     model.train()
#     total_loss, correct, total = 0, 0, 0
#
#     for X_batch, y_batch, _ in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#
#         logits = model(X_batch)
#         loss = criterion(logits, y_batch)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#         pred = torch.argmax(logits, dim=1)
#         correct += (pred == y_batch).sum().item()
#         total += y_batch.size(0)
#
#     acc = correct / total * 100
#     print(f" Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.2f}%")
#
# # -----------------------------
# #  예측 결과 비교
# # -----------------------------
# print("\n 예측 결과 (GT vs Pred)")
# model.eval()
# with torch.no_grad():
#     for X, y, vid in eval_loader:
#         X = X.to(device)
#         logits = model(X)
#         pred = torch.argmax(logits, dim=1).item()
#         gt = y.item()
#         status = "✅" if pred == gt else "❌"
#         print(f"▶ 영상: {vid[0]} / GT: {gt} / Pred: {pred} {status}")
from pathlib import Path
import numpy as np

# 경로 설정
DATA_ROOT = Path("C:/Users/user/Desktop/skh/ETRI")
GZ_PATH = DATA_ROOT / "processed_videos"
INFO_PATH = DATA_ROOT / "PNU_selected"

# # 테스트 대상 subject 목록
# test_subjects = {
#     "A2-1022-5", "A2-1026-3", "A2-1026-04", "A2-1029-02", "A2-1029-03", "A2-1029-04", "A2-1029-05"
# }

# event_info.txt에서 라벨 파싱
def parse_event_label(event_info_path):
    try:
        with open(event_info_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                clean = line.split("#")[0].strip()
                if clean.isdigit():
                    return int(clean)
    except Exception as e:
        print(f"❌ label 로딩 실패 ({event_info_path}): {e}")
    return None

# 데이터 초기화
X_train, y_train, id_train = [], [], []
X_test, y_test, id_test = [], [], []

# 모든 event_info.txt 탐색
for event_info_path in INFO_PATH.rglob("event_info.txt"):
    try:
        event_id = event_info_path.parent.name  # 예: 000, 001
        label = parse_event_label(event_info_path)
        if label is None:
            continue

        # 경로에서 subject_name, session_id 추출
        parts = event_info_path.parts
        subject_name = parts[parts.index("PNU_selected") + 1]
        session_id = parts[parts.index(subject_name) + 1]

        print(f"\n📄 subject: {subject_name}, session: {session_id}, event_id: {event_id} → label = {label}")

        expected_prefix = f"processed_{subject_name}_{session_id}_rec_{event_id}_"
        matched_folders = [
            f for f in GZ_PATH.iterdir()
            if f.name.startswith(expected_prefix) and f.name.endswith("_out_resnet_x128_vertex")
        ]

        if not matched_folders:
            print(f"🚫 No matched folder for {expected_prefix}")
            continue

        for folder in sorted(matched_folders):
            gaze_txt_path = folder / "predicted_gaze_vectors.txt"
            if not gaze_txt_path.exists():
                continue


            try:
                gaze_seq = np.loadtxt(gaze_txt_path, delimiter=',')
                MAX_LEN = 300
                if gaze_seq.shape[0] > MAX_LEN:
                    print(f"⚠ 프레임 수 초과: {gaze_seq.shape[0]}")
                    continue

                padded = np.zeros((MAX_LEN, 3), dtype=np.float32)
                padded[:gaze_seq.shape[0]] = gaze_seq

                # 학습 / 테스트 분리
                if label in [0, 1]:
                    target_list = X_test if subject_name in test_subjects else X_train
                    label_list = y_test if subject_name in test_subjects else y_train
                    id_list    = id_test if subject_name in test_subjects else id_train

                    target_list.append(padded)
                    label_list.append(label)
                    id_list.append(folder.name)
                    print(f"📌 저장 완료 → {folder.name}")

            except Exception as e:
                print(f"❌ gaze vector 로딩 실패: {e}")

    except Exception as e:
        print(f"⚠️ 예외 발생: {e}")

print(f"\n✅ 학습 샘플 수: {len(X_train)}, 테스트 샘플 수: {len(X_test)}")
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


print(f"✅ 최종 학습 샘플 수: {len(X_train)} / 라벨 종류: {set(y_train)}")
print(f"✅ 최종 테스트 샘플 수: {len(X_test)} / 라벨 종류: {set(y_test)}")


# -----------------------------
#  Dataset 구성
# -----------------------------
class GazeDataset(Dataset):
    def __init__(self, X, y, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.ids = ids  # 문자열 리스트

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ids[idx]

# train_loader: 학습용 / eval_loader: 순서 고정된 평가용
X_train = np.array(X_train)
y_train = np.array(y_train)
id_train = np.array(id_train)

X_test = np.array(X_test)
y_test = np.array(y_test)
id_test = np.array(id_test)

train_dataset = GazeDataset(X_train, y_train, id_train)
test_dataset = GazeDataset(X_test, y_test, id_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



# -----------------------------
#  LSTM 모델 정의
# -----------------------------
class GazeLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)               # out: [batch, seq_len, hidden_dim]
        final_feature = out.mean(dim=1)     # 전체 시점 평균
        return self.fc(final_feature)


# -----------------------------
#  학습 루프
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch, _ in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total * 100
    print(f" Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.2f}%")

# -----------------------------
#  예측 결과 비교
# -----------------------------
print("\n 예측 결과 (GT vs Pred)")
model.eval()
with torch.no_grad():
    for X, y, vid in eval_loader:
        video_name = vid[0]  # 문자열로 변환된 폴더 이름

        # ✅ 특정 이름만 필터링
        if not (
                "000_000036592912" in video_name or
                "001_000336294412" in video_name
        ):
            continue  # 나머지는 스킵
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).item()
        gt = y.item()
        status = "✅" if pred == gt else "❌"
        print(f"▶ 영상: {vid[0]} / GT: {gt} / Pred : {pred} {status}")