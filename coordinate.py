# from pathlib import Path
# import numpy as np
# import pandas as pd
# import cv2
# from matplotlib.path import Path as MplPath
# import torch
# import torch.backends.cudnn as cudnn
# import os
# from utils import config as cfg, update_config
# from models import FaceDetectorIF as FaceDetector
# from models import GazePredictorHandler as GazePredictor
#
# # ⚙️ config 설정
# config_path = "C:/Users/user/Desktop/skh/3DGazeNet/demo/configs/infer_res18_x128_all_vfhq_vert.yaml"
# update_config(config_path)
#
# # ✅ 모델 초기화
# detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
# predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)
#
# # 📁 루트 경로
# root_path = Path("C:/Users/user/Desktop/skh/ETRI/processed_videos")
#
# # 🎯 모니터 키워드 및 좌표
# right_monitor_keywords = ["000_000741294612", "000_000036592912"]
# left_monitor_keywords = ["001_000137300112", "001_000336294412"]
# right_monitor_coords = np.array([(565, 84), (639, 45), (639, 195), (560, 203)])
# left_monitor_coords = np.array([(1, 24), (138, 112), (157, 211), (1, 190)])
#
# # 🧭 헬퍼 함수
# def point_in_quad(point, quad):
#     return MplPath(quad).contains_point(point)
#
# def compute_endpoint(origin, gaze, scale=150):
#     return (origin[0] + gaze[0] * scale, origin[1] + gaze[1] * scale)
#
# def load_origin_list(origin_file):
#     if not origin_file.exists():
#         return None
#     origins = []
#     with open(origin_file, "r") as f:
#         for line in f:
#             try:
#                 x, y = line.strip().split(",")
#                 origins.append((float(x), float(y)))
#             except:
#                 origins.append((np.nan, np.nan))
#     return origins
#
# # 📊 결과 저장용 리스트
# results = []
#
# # 🔍 폴더 순회
# for folder in root_path.iterdir():
#     if not folder.is_dir() or "_02_rec_" not in folder.name:
#         continue
#
#     folder_name = folder.name
#     if any(k in folder_name for k in right_monitor_keywords):
#         monitor_quad = right_monitor_coords
#     elif any(k in folder_name for k in left_monitor_keywords):
#         monitor_quad = left_monitor_coords
#     else:
#         continue
#
#     video_path = folder / "output.mp4"
#     gaze_path = folder / "predicted_gaze_vectors.txt"
#     origin_path = folder / "origin_points.txt"  # ← 여기서 origin 좌표 사용
#
#     if not gaze_path.exists() or not video_path.exists() or not origin_path.exists():
#         print(f"[스킵] 파일 없음: {folder_name}")
#         continue
#
#     try:
#         gaze_vectors = np.loadtxt(gaze_path, delimiter=',')
#         if len(gaze_vectors.shape) == 1:
#             gaze_vectors = gaze_vectors[np.newaxis, :]
#
#         origin_list = load_origin_list(origin_path)
#         if origin_list is None or len(origin_list) != len(gaze_vectors):
#             print(f"[스킵] origin 길이 불일치: {folder_name}")
#             continue
#
#         cap = cv2.VideoCapture(str(video_path))
#         total, inside = 0, 0
#
#         for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
#             ret, frame = cap.read()
#             if not ret or i >= len(gaze_vectors):
#                 break
#
#             origin = origin_list[i]
#             if np.isnan(origin[0]):
#                 continue
#
#             gaze = gaze_vectors[i][:3]
#             end = compute_endpoint(origin, gaze[:2])
#             if point_in_quad(end, monitor_quad):
#                 inside += 1
#             total += 1
#
#         cap.release()
#         ratio = inside / total if total else 0
#         label = 0 if ratio >= 0.5 else 1
#         results.append((folder_name, ratio, label))
#         print(f"{folder_name}: {'✅ 정반응' if label == 0 else '❌ 오반응'} ({ratio:.2f})")
#
#     except Exception as e:
#         print(f"❌ 처리 실패: {folder_name} - {e}")
#
# # 💾 CSV 저장
# df = pd.DataFrame(results, columns=["folder", "in_ratio", "label"])
# df.to_csv(root_path / "gaze_evaluation_results.csv", index=False)
# print("\n✅ 평가 결과 저장 완료 → gaze_evaluation_results.csv")



#
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from matplotlib.path import Path as MplPath
# import cv2
#
# # 📂 평가 대상 폴더
# root_path = Path("C:/Users/user/Desktop/skh/ETRI/processed_videos")
#
# # 📌 모니터 좌표
# right_monitor_keywords = ["000_000741294612", "000_000036592912"]
# left_monitor_keywords = ["001_000137300112", "001_000336294412"]
#
# right_monitor_coords = np.array([(565, 84), (639, 45), (639, 195), (560, 203)])
# left_monitor_coords = np.array([(1, 24), (138, 112), (157, 211), (1, 190)])
#
# def point_in_quad(point, quad):
#     return MplPath(quad).contains_point(point)
#
# def compute_endpoint(origin, gaze, scale=150):
#     return (origin[0] + gaze[0] * scale, origin[1] + gaze[1] * scale)
#
# results = []
#
# for folder in root_path.iterdir():
#     if not folder.is_dir() or "_02_rec_" not in folder.name:
#         continue
#
#     folder_name = folder.name
#
#     # 🎯 모니터 방향 판단
#     if any(k in folder_name for k in right_monitor_keywords):
#         monitor_quad = right_monitor_coords
#     elif any(k in folder_name for k in left_monitor_keywords):
#         monitor_quad = left_monitor_coords
#     else:
#         continue
#
#     gaze_path = folder / "predicted_gaze_vectors.txt"
#     origin_path = folder / "origin_points.txt"
#     video_path = folder / "output.mp4"
#
#     if not gaze_path.exists() or not origin_path.exists() or not video_path.exists():
#         print(f"[스킵] 파일 없음: {folder_name}")
#         continue
#
#     try:
#         gaze_vectors = np.loadtxt(gaze_path, delimiter=',')
#         origin_points = np.loadtxt(origin_path, delimiter=',')
#
#         if gaze_vectors.ndim == 1:
#             gaze_vectors = gaze_vectors[np.newaxis, :]
#         if origin_points.ndim == 1:
#             origin_points = origin_points[np.newaxis, :]
#
#         if len(gaze_vectors) != len(origin_points):
#             print(f"[경고] 프레임 수 불일치: gaze={len(gaze_vectors)}, origin={len(origin_points)} in {folder_name}")
#
#         total, inside = 0, 0
#         valid = 0
#
#         for i in range(min(len(gaze_vectors), len(origin_points))):
#             gaze = gaze_vectors[i][:3]
#             origin = origin_points[i][:2]
#
#             if np.isnan(origin[0]) or np.isnan(origin[1]):
#                 continue
#
#             end = compute_endpoint(origin, gaze[:2])
#             if point_in_quad(end, monitor_quad):
#                 inside += 1
#             total += 1
#             valid += 1
#
#         ratio = inside / total if total else 0
#         label = 0 if ratio >= 0.5 else 1
#         results.append((folder_name, ratio, label))
#         print(f"{folder_name}: {'✅ 정반응' if label == 0 else '❌ 오반응'} → label={label}, in_ratio={ratio:.2f}, valid_frames={valid}")
#
#     except Exception as e:
#         print(f"❌ 오류 발생: {folder_name} - {e}")
#
# # 💾 결과 저장
# df = pd.DataFrame(results, columns=["folder", "in_ratio", "label"])
# df.to_csv(root_path / "gaze_evaluation_results.csv", index=False)
# print("\n✅ 평가 완료: gaze_evaluation_results.csv 생성됨")
import cv2
import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath
from matplotlib.path import Path as MplPath
import numpy as np



# 📍 설정
video_path = Path("C:/Users/user/Desktop/skh/ETRI/processed_videos/processed_221010_001_02_rec_000_000741294612_out_resnet_x128_vertex/output.mp4")
gaze_path = video_path.parent / "predicted_gaze_vectors.txt"

# 💻 모니터 영역 (수정 가능)
monitor_coords = np.array([(565, 84), (639, 45), (639, 195), (560, 203)])
monitor_polygon = MplPath(monitor_coords)

# 🎯 시선 벡터 불러오기
gaze_vectors = np.loadtxt(gaze_path, delimiter=',')
if gaze_vectors.ndim == 1:
    gaze_vectors = gaze_vectors[np.newaxis, :]

# 🎥 영상 준비
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = video_path.parent / "visualized_center_based.mp4"
out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

scale = 400  # 벡터 길이 조절
origin_point = (w // 2, h // 2)  # 화면 중앙 기준

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= len(gaze_vectors):
        break

    gaze = gaze_vectors[frame_idx][:2]
    if not np.isnan(gaze[0]):
        end_point = (
            int(origin_point[0] + gaze[0] * scale),
            int(origin_point[1] + gaze[1] * scale)
        )

        color = (0, 255, 0) if monitor_polygon.contains_point(end_point) else (0, 0, 255)
        cv2.arrowedLine(frame, origin_point, end_point, color, 2, tipLength=0.2)

    # 🖼 모니터 영역 표시
    for i in range(4):
        pt1 = tuple(monitor_coords[i])
        pt2 = tuple(monitor_coords[(i + 1) % 4])
        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

    out.write(frame)
    frame_idx += 1
def is_inside_monitor_box(origin, gaze_vector, monitor_coords, scale=400):
    """origin 기준 gaze_vector가 monitor 영역 안에 있는지 판단"""
    end_point = (
        origin[0] + gaze_vector[0] * scale,
        origin[1] + gaze_vector[1] * scale
    )
    polygon = MplPath(monitor_coords)
    return polygon.contains_point(end_point)
cap.release()
out.release()
print(f"✅ 시각화 완료: {out_path}")
