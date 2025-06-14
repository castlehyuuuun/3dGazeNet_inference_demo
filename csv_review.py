# import os
# import re
# import csv
#
# base_dir = r'C:/Users/user/Desktop/skh/ETRI/processed_videos'
# output_csv = 'gaze_summary_metrics.csv'
#
# # 결과 저장 리스트
# results = []
#
# # 대상 폴더만 필터링
# for folder in os.listdir(base_dir):
#     if folder.endswith('_out_resnet_x128_vertex'):
#         if 'A2-1022-2' in folder and 'rec_001' in folder:
#             continue
#         folder_path = os.path.join(base_dir, folder)
#         summary_path = os.path.join(folder_path, 'gaze_inside_monitor_summary.txt')
#
#         if os.path.exists(summary_path):
#             with open(summary_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#
#             # 첫 줄: 전체 프레임 수, 정반응 수, 비율
#             match_total = re.search(r'전체 프레임 수:\s*(\d+), 정반응 수:\s*(\d+), 비율:\s*([\d.]+)', lines[0])
#             total_frames, correct_count, ratio = match_total.groups() if match_total else ('', '', '')
#
#             # 다음 줄들: 프레임 기준별 결과
#             frame_results = {}
#             for line in lines[1:]:
#                 match_frame = re.search(r'(\d+)프레임 기준 → (정반응|오반응)', line)
#                 if match_frame:
#                     frame, result = match_frame.groups()
#                     frame_results[frame] = result
#
#             results.append([
#                 folder,
#                 total_frames,
#                 correct_count,
#                 ratio,
#                 frame_results.get('30', ''),
#                 frame_results.get('60', ''),
#                 frame_results.get('90', ''),
#                 frame_results.get('120', '')
#             ])
#
# # CSV로 저장
# with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['폴더명', '전체프레임수', '정반응수', '비율', '30프레임', '60프레임', '90프레임', '120프레임'])
#     writer.writerows(results)
# output_csv = os.path.join(base_dir, 'gaze_summary_metrics.csv')
#
# print(f'결과 저장 완료: {output_csv}')


import os
import re
import csv

base_dir = r'C:/Users/user/Desktop/skh/ETRI/processed_videos'
output_csv = os.path.join(base_dir, 'gaze_summary_metrics.csv')  # 저장 경로 명시

# 결과 저장 리스트
results = []

for folder in os.listdir(base_dir):
    if folder.endswith('_out_resnet_x128_vertex'):
        if 'A2-1022-2' in folder and 'rec_001' in folder:
            continue

        folder_path = os.path.join(base_dir, folder)
        summary_path = os.path.join(folder_path, 'gaze_inside_monitor_summary.txt')

        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 전체 프레임 / 정반응 수 / 비율
            # match_total = re.search(r'전체 프레임 수:\s*(\d+), 정반응 수:\s*(\d+), 비율:\s*([\d.]+)', lines[0])
            # total_frames, correct_count, ratio = match_total.groups() if match_total else ('', '', '')

            total_frames = correct_count = ratio = ''
            for line in lines:
                if '전체 프레임 수' in line:
                    match = re.search(r'전체 프레임 수:\s*(\d+)', line)
                    if match:
                        total_frames = match.group(1)
                if '정반응 수' in line:
                    match = re.search(r'정반응 수:\s*(\d+)', line)
                    if match:
                        correct_count = match.group(1)
                if '비율' in line:
                    match = re.search(r'비율:\s*([\d.]+)', line)
                    if match:
                        ratio = match.group(1)

            # rec_000, rec_001 그룹 정보 추출
            rec_match = re.search(r'(rec_0\d\d)', folder)
            rec = rec_match.group(1) if rec_match else ''

            # 프레임 기준별 판정 추출
            frame_results = {}
            for line in lines[1:]:
                match_frame = re.search(r'(\d+)프레임 기준 → (정반응|오반응)', line)
                if match_frame:
                    frame, result = match_frame.groups()
                    frame_results[frame] = result

            results.append([
                folder,
                rec,
                total_frames,
                correct_count,
                ratio,
                frame_results.get('30', ''),
                frame_results.get('60', ''),
                frame_results.get('90', ''),
                frame_results.get('120', '')
            ])

# CSV로 저장
with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['폴더명', 'rec', '전체프레임수', '정반응수', '비율', '30프레임', '60프레임', '90프레임', '120프레임'])
    writer.writerows(results)

print(f'✅ 결과 저장 완료: {output_csv}')
