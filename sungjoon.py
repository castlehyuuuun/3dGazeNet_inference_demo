# import numpy as np
# import cv2
# import os
# import shutil
# import pandas as pd
# import matplotlib.pyplot as plt
# def to_frame(video_path):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     ret = True
#     num = 0
#     while ret:
#         ret, img = cap.read()
#         if ret:
#             frames.append(img)
#             cv2.imwrite(f'{img_temp}frame_{num}.png', img)
#             num += 1
#     cap.release()
#     return frames
#
#
# def get_max_coord_from_frame(frame):
#     """
#     frame: NumPy 배열 형태의 프레임 (H, W, 3) - 컬러 이미지
#     return: (x, y) - 가장 주시한 좌표
#     """
#     if frame.ndim == 3 and frame.shape[2] == 3:
#         # BGR to Grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     elif frame.ndim == 2:
#         gray = frame  # 이미 그레이스케일인 경우
#     else:
#         raise ValueError("입력은 (H, W, 3) 또는 (H, W) 형태여야 합니다.")
#
#     # 최댓값 좌표 찾기
#     max_y, max_x = np.unravel_index(np.argmax(gray), gray.shape)
#     return max_x, max_y
# def gaze_on_bbox_with_multiple_scales(cx, cy, dx, dy, bbox, num_steps=10):
#     x1, y1, x2, y2 = bbox
#     dx, dy = float(dx), float(dy)
#     norm = np.sqrt(dx**2 + dy**2)
#     if norm == 0:
#         return False
#     dx /= norm
#     dy /= norm
#
#     # 얼굴 중심에서 bbox x1, x2까지 거리
#     dist1 = abs(cx - x1)
#     dist2 = abs(cx - x2)
#
#     # x1과 x2 사이의 거리 범위
#     min_dist = min(dist1, dist2)
#     max_dist = max(dist1, dist2)
#
#     # 여러 scale 값 생성
#     scales = np.linspace(min_dist, max_dist, num_steps)
#
#     # 각 scale에 대해 gaze vector 끝점 계산 후 bbox 내부 여부 확인
#     for scale in scales:
#         ex = cx + dx * scale
#         ey = cy + dy * scale
#         if x1 <= ex <= x2 and y1 <= ey <= y2:
#             return True
#     return False
# def is_in_bbox(x, y, bbox):
#     x1, y1, x2, y2 = bbox
#     return x1 <= x <= x2 and y1 <= y <= y2
# def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
#     # 두 선분 (x1,y1)-(x2,y2), (x3,y3)-(x4,y4) 교차 여부
#     denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
#     if denom == 0:
#         return False
#     ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
#     ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
#     return 0 <= ua <= 1 and 0 <= ub <= 1
#
# def gaze_hits_box(cx, cy, dx, dy, box, scale=2000):
#     # box: (left, top, right, bottom)
#     x1, y1 = cx, cy
#     x2, y2 = cx + dx*scale, cy + dy*scale
#     left, top, right, bottom = box
#     # box 네 변
#     edges = [
#         (left, top, right, top),       # 상
#         (right, top, right, bottom),   # 우
#         (right, bottom, left, bottom), # 하
#         (left, bottom, left, top),     # 좌
#     ]
#     for ex1, ey1, ex2, ey2 in edges:
#         if line_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
#             return True
#     return False
# if __name__ == "__main__":
#     # data_folder = rf'C:/Users/tjdwn/Desktop/ETRI 2025/DB/PNU_selected'
#     data_folder = rf"./results"
#     kids = os.listdir(data_folder)
#     act_1 = ['000_000741294612.mp4', '000_000036592912.mp4']
#     act_2 = ['001_000137300112.mp4', '001_000336294412.mp4']
#     act_1_bbox = [[1780, 265, 2048, 655],[1780, 310, 2048, 680]]
#     act_2_bbox = [[0, 245, 450, 670], [0, 330, 330, 700]]
#     for kid in kids:
#         # if kid == 'A2-1022-2':
#         #     continue
#         for scenario in os.listdir(os.path.join(data_folder, kid)):
#             if scenario not in ['02']:
#                 continue
#             if scenario in ['02']:
#                 for video in os.listdir(os.path.join(data_folder, kid, scenario, 'rec')):
#                     if video not in act_1:
#                         continue
#                     if video == act_1[0]:
#                         bbox = act_1_bbox[0]
#                     else:
#                         bbox = act_1_bbox[1]
#                     original_vid_path_ = os.path.join(data_folder, kid, scenario, 'rec', video)
#
#                     print(original_vid_path_)
#                     label_temp = './temp/'
#                     img_temp = os.path.join(label_temp, 'img/')
#                     save_path = os.path.join('C:/Users/tjdwn/PycharmProjects/pythonProject7/gazelle/tv_object_detection', kid, scenario, 'rec', 'frame_act2/')
#                     save_vid_path = os.path.join('C:/Users/tjdwn/PycharmProjects/pythonProject7/gazelle/tv_object_detection', kid, scenario, 'rec', video)
#                     txt_path = os.path.join('/home/aivs/바탕화면/hdd/ETRI 2025/ETRI/L2CS-Net/results', kid, scenario,'rec')
#                     result = 0
#                     if os.path.isdir(label_temp):
#                         shutil.rmtree(label_temp)
#                     os.makedirs(save_path, exist_ok=True)
#                     os.makedirs(img_temp, exist_ok=True)
#
#                     frames = to_frame(original_vid_path_)
#                     x1, y1, x2, y2 = bbox
#                     column_names = ['x1', 'y1', 'x2', 'y2', 'dx', 'dy']
#                     df = pd.read_csv(f'{txt_path}/{video}.txt', names=column_names)
#                     bc = (bbox[0] + bbox[2]) / 2
#                     for i, frame in enumerate(frames):
#                         if df.iloc[i]['x1'] == 0 and df.iloc[i]['x2'] == 0:
#                             cx, cy = 0, 0
#                         else:
#                             cx = (df.iloc[i]['x1'] + df.iloc[i]['x2']) / 2
#                             cy = (df.iloc[i]['y1'] + df.iloc[i]['y2']) / 2
#                         # if cx > bc:
#                         #     scale = cx - bc
#                         # else:
#                         #     scale = bc - cx
#
#                         dx, dy = -df.iloc[i]['dx'], -df.iloc[i]['dy']
#                         # norm = np.sqrt(dx**2 + dy**2)
#                         # dx = dx / norm
#                         # dy = dy / norm
#                         # print(dx, dy)
#                         pred = gaze_on_bbox_with_multiple_scales(cx, cy, dx, dy, bbox, num_steps=500)
#                         # ex = cx + dx * scale
#                         # ey = cy + dy * scale
#                         # print(ex, ey)
#                         #
#                         # if (x1 <= ex <= x2) and (y1 <= ey <= y2):
#                         #     pred = 1
#                         # else:
#                         #     pred = 0
#
#
#
#                         # roi = frame[y1:y2, x1:x2, :]
#                         #
#                         # colormap = plt.cm.get_cmap('jet', 256)
#                         # color_table = (colormap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)  # (256,3)
#                         #
#                         #
#                         #
#                         #
#                         # # roi가 (H,W,3)일 때
#                         # heat_vals = np.zeros(roi.shape[:2])
#                         # for i in range(roi.shape[0]):
#                         #     for j in range(roi.shape[1]):
#                         #         heat_vals[i, j] = rgb2heatval(roi[i, j])
#
#
#                         if pred:
#                             result += 1
#                             print("정반응")
#                         else:
#                             print("미반응")
#
#                         # if np.max(heat_vals) > 0.895:
#                         #     result += 1
#                         #     print(np.max(heat_vals))
#                         #     print("정반응")
#                         # else:
#                         #     print(np.max(heat_vals))
#                         #     print("미반응")
#                     txt = open(f'result_gray_act1.txt', 'a+')
#                     txt.write(f"{kid}, %d\n" % (result))
#                     txt.close()

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt

column_names = ['kid_name', 'result']
df = pd.read_csv('./result_gray_act2.txt', names=column_names)
gt = pd.read_csv('./act_2_gt.txt', names=['result'])
gt.index = df.index

thresholds = list(range(1, 151)) # 0, 30, ..., 150
f1_scores = []
accuracy_scores = []
recall_scores = []
precision_scores = []

for threshold in thresholds:
    y_true = []
    y_pred = []
    for i in df.index:
        result = int(df.loc[i, 'result'])
        grt = int(gt.loc[i])
        pred = 0 if result >= threshold else 1
        y_true.append(grt)
        y_pred.append(pred)
    f1 = f1_score(y_true, y_pred) * 100
    acc = accuracy_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    f1_scores.append(f1)
    accuracy_scores.append(acc)
    recall_scores.append(recall)
    precision_scores.append(precision)

# 결과 출력
for threshold, f1, acc, recall, precision in zip(thresholds, f1_scores, accuracy_scores, recall_scores, precision_scores):
    print(f"Threshold {threshold}: F1 Score = {f1:.2f}%, Accuracy = {acc:.2f}%, Recall = {recall:.2f}%, Precision = {precision:.2f}%")

# 선 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1 Score (%)')
plt.plot(thresholds, accuracy_scores, label='Accuracy (%)')
plt.plot(thresholds, recall_scores, label='Recall (%)')
plt.plot(thresholds, precision_scores, label='Precision (%)')
plt.xlabel('Threshold')
plt.ylabel('Score (%)')
plt.title('SpatioTemporal Action2 Result')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 151, 30))  # x축을 30 단위로
plt.savefig('./result_act2.png')
plt.show()

