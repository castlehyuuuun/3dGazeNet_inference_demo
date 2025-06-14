# import cv2
# import torch
# import numpy as np
# from pathlib import Path
# from matplotlib.path import Path as MplPath
# from models import FaceDetectorIF as FaceDetector
# from models import GazePredictorHandler as GazePredictor
# from utils import config as cfg, update_config
#
# # ✅ 설정값 로딩
# update_config("C:/Users/user/Desktop/skh/3DGazeNet/demo/configs/infer_res18_x128_all_vfhq_vert.yaml")
#
# device = cfg.DEVICE
# face_detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
# gaze_predictor = GazePredictor(cfg.PREDICTOR, device=device)
#
# # ✅ 모니터 좌표 및 대상 키워드
# right_monitor_keywords = ["000_000741294612", "000_000036592912"]
# left_monitor_keywords = ["001_000137300112", "001_000336294412"]
#
# right_monitor_coords = np.array([(565, 84), (639, 45), (639, 195), (560, 203)])
# left_monitor_coords = np.array([(1, 24), (138, 112), (157, 211), (1, 190)])
#
#
# def point_in_quad(point, quad):
#     return MplPath(quad).contains_point(point)
#
# def compute_endpoint(origin, gaze, scale=400):
#     return (origin[0] + gaze[0] * scale, origin[1] + gaze[1] * scale)
#
# # ✅ 영상 루프
# root_path = Path("C:/Users/user/Desktop/skh/ETRI/processed_videos")
# for folder in root_path.iterdir():
#     if not folder.is_dir() or "_02_rec_" not in folder.name:
#         continue
#
#     if any(k in folder.name for k in right_monitor_keywords):
#         monitor_quad = right_monitor_coords
#     elif any(k in folder.name for k in left_monitor_keywords):
#         monitor_quad = left_monitor_coords
#     else:
#         continue
#
#     video_path = folder / "output.mp4"
#     if not video_path.exists():
#         print(f"[스킵] 영상 없음: {folder.name}")
#         continue
#
#     cap = cv2.VideoCapture(str(video_path))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out_path = folder / "visualized.mp4"
#     out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#
#     total, inside = 0, 0
#     frame_idx = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 얼굴 검출 및 시선 예측
#         face_info = face_detector.infer(frame)
#         if not face_info:
#             out.write(frame)
#             frame_idx += 1
#             continue
#
#         out_dict = gaze_predictor.infer(frame, face_info)
#         gaze = out_dict["gaze"][:2]
#         origin = out_dict["lms5"][0][:2]  # 왼쪽 눈 기준
#
#         if np.isnan(origin[0]) or np.isnan(gaze[0]):
#             out.write(frame)
#             frame_idx += 1
#             continue
#
#         end = compute_endpoint(origin, gaze)
#         color = (0, 255, 0) if point_in_quad(end, monitor_quad) else (0, 0, 255)
#         if point_in_quad(end, monitor_quad):
#             inside += 1
#         total += 1
#
#         # 시선 벡터 시각화
#         cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(np.array(end).astype(int)), color, 2, tipLength=0.2)
#
#         # 모니터 사각형 그리기
#         for i in range(4):
#             pt1 = tuple(monitor_quad[i])
#             pt2 = tuple(monitor_quad[(i + 1) % 4])
#             cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
#
#         out.write(frame)
#         frame_idx += 1
#
#     cap.release()
#     out.release()
#
#     ratio = inside / total if total else 0
#     label = 0 if ratio >= 0.5 else 1
#     print(f"{folder.name}: {'✅ 정반응' if label == 0 else '❌ 오반응'} → ratio={ratio:.2f}, label={label}")

import cv2
import numpy as np
import os

# 수정: 영상 경로 설정
video_path = "C:/Users/user/Desktop/skh/ETRI/processed_videos/processed_A2-1019-4_02_rec_001_000336294412_out_resnet_x128_vertex/output.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 영상을 불러올 수 없습니다.")
    exit()

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"📌 Point {len(points)}: ({x}, {y})")

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (255, 0, 0), 2)
        if len(points) == 4:
            cv2.line(frame, points[-1], points[0], (255, 0, 0), 2)
        cv2.imshow("Click 4 corners", frame)

# 창 설정
cv2.namedWindow("Click 4 corners", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Click 4 corners", 960, 540)
cv2.setMouseCallback("Click 4 corners", mouse_callback)
cv2.imshow("Click 4 corners", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    coords = np.array(points)
    print("✅ 추출된 좌표:", coords)

    # 저장 경로 설정 (예: 영상이름 기준 저장)
    save_name = os.path.basename(video_path).replace(".mp4", "_monitor_coords.npy")
    save_path = os.path.join(os.path.dirname(video_path), save_name)
    np.save(save_path, coords)
    print(f"💾 좌표 저장 완료: {save_path}")
else:
    print("❌ 4개의 좌표가 선택되지 않았습니다.")
