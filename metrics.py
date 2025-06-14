# # import numpy as np
# # import pandas as pd
# # from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
# # import matplotlib.pyplot as plt
# #
# # column_names = ['kid_name', 'result']
# # df = pd.read_csv('./result_gray_act2.txt', names=column_names)
# # gt = pd.read_csv('./act_2_gt.txt', names=['result'])
# # gt.index = df.index
# #
# # thresholds = list(range(1, 151)) # 0, 30, ..., 150
# # f1_scores = []
# # accuracy_scores = []
# # recall_scores = []
# # precision_scores = []
# #
# # for threshold in thresholds:
# #     y_true = []
# #     y_pred = []
# #     for i in df.index:
# #         result = int(df.loc[i, 'result'])
# #         grt = int(gt.loc[i])
# #         pred = 0 if result >= threshold else 1
# #         y_true.append(grt)
# #         y_pred.append(pred)
# #     f1 = f1_score(y_true, y_pred) * 100
# #     acc = accuracy_score(y_true, y_pred) * 100
# #     recall = recall_score(y_true, y_pred) * 100
# #     precision = precision_score(y_true, y_pred) * 100
# #     f1_scores.append(f1)
# #     accuracy_scores.append(acc)
# #     recall_scores.append(recall)
# #     precision_scores.append(precision)
# #
# # # 결과 출력
# # for threshold, f1, acc, recall, precision in zip(thresholds, f1_scores, accuracy_scores, recall_scores, precision_scores):
# #     print(f"Threshold {threshold}: F1 Score = {f1:.2f}%, Accuracy = {acc:.2f}%, Recall = {recall:.2f}%, Precision = {precision:.2f}%")
# #
# # # 선 그래프 그리기
# # plt.figure(figsize=(10, 6))
# # plt.plot(thresholds, f1_scores, label='F1 Score (%)')
# # plt.plot(thresholds, accuracy_scores, label='Accuracy (%)')
# # plt.plot(thresholds, recall_scores, label='Recall (%)')
# # plt.plot(thresholds, precision_scores, label='Precision (%)')
# # plt.xlabel('Threshold')
# # plt.ylabel('Score (%)')
# # plt.title('SpatioTemporal Action2 Result')
# # plt.legend()
# # plt.grid(True)
# # plt.xticks(np.arange(0, 151, 30))  # x축을 30 단위로
# # plt.savefig('./result_act2.png')
# # plt.show()

#
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
#
# # 데이터 불러오기
# df = pd.read_csv('C:/Users/user/Desktop/skh/ETRI/processed_videos/gaze_summary_metrics.csv')
# df["전체프레임수"] = pd.to_numeric(df["전체프레임수"], errors="coerce")
# df["정반응수"] = pd.to_numeric(df["정반응수"], errors="coerce")
# df = df.dropna(subset=["전체프레임수", "정반응수", "rec"])
#
# # 평가 프레임 기준
# # threshold_frames = ["30", "60", "90", "120"]
# threshold_frames = list(range(1, 151))
# results = []
#
# # 평가지표 계산
# for rec_group in ["rec_000", "rec_001"]:
#     sub_df = df[df["rec"] == rec_group]
#     for th in threshold_frames:
#         threshold = int(th)
#         y_true = (sub_df["정반응수"] >= sub_df["전체프레임수"] / 2).astype(int).map({1: 0, 0: 1})
#         y_pred = (sub_df["정반응수"] >= threshold).astype(int).map({1: 0, 0: 1})
#
#         f1 = f1_score(y_true, y_pred, zero_division=0) * 100
#         acc = accuracy_score(y_true, y_pred) * 100
#         rec = recall_score(y_true, y_pred, zero_division=0) * 100
#         prec = precision_score(y_true, y_pred, zero_division=0) * 100
#
#         results.append({
#             "rec_group": rec_group,
#             "threshold": threshold,
#             "F1 Score": f1,
#             "Accuracy": acc,
#             "Recall": rec,
#             "Precision": prec
#         })
#
# # 데이터프레임 생성
# result_df = pd.DataFrame(results)
#
# # 📊 그룹별로 그래프 따로 저장
#
# # rec 이름을 act 이름으로 매핑
# rec_to_act = {
#     "rec_000": "act1",
#     "rec_001": "act2"
# }
#
# metrics = ["F1 Score", "Accuracy", "Recall", "Precision"]
# for rec_group in ["rec_000", "rec_001"]:
#     act_group = rec_to_act[rec_group]
#     plt.figure(figsize=(8, 6))
#     subset = result_df[result_df["rec_group"] == rec_group]
#     for metric in metrics:
#         plt.plot(subset["threshold"], subset[metric], marker='o', label=metric)
#     plt.title(f"{act_group} Metrics")
#     plt.xlabel("Threshold (the number of thresholds)")
#     plt.ylabel("Score (%)")
#     plt.xticks(subset["threshold"])
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     save_path = f"eval_metrics_{rec_group}.png"
#     plt.savefig(save_path)
#     plt.show()

#
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
# import platform
# import re
#
# # 폰트 설정
# if platform.system() == 'Windows':
#     plt.rcParams['font.family'] = 'Malgun Gothic'
# elif platform.system() == 'Darwin':
#     plt.rcParams['font.family'] = 'AppleGothic'
# else:
#     plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 데이터 불러오기
# df = pd.read_csv('C:/Users/user/Desktop/skh/ETRI/processed_videos/gaze_summary_metrics.csv')
# df["전체프레임수"] = pd.to_numeric(df["전체프레임수"], errors="coerce")
# df["정반응수"] = pd.to_numeric(df["정반응수"], errors="coerce")
#
# if "rec" not in df.columns:
#     df["rec"] = df["폴더명"].str.extract(r"(rec_0\d\d)")
# df = df.dropna(subset=["전체프레임수", "정반응수", "rec"])
#
# # 프레임 기준 및 강조 기준
# threshold_frames = list(range(1, 151))
# highlight_thresholds = [30, 60, 90, 120]
#
# # rec → act 매핑
# rec_to_act = {
#     "rec_000": "act1",
#     "rec_001": "act2"
# }
#
# results = []
#
# # 평가지표 계산
# for rec_group in ["rec_000", "rec_001"]:
#     sub_df = df[df["rec"] == rec_group]
#     for threshold in threshold_frames:
#         y_true = (sub_df["정반응수"] >= sub_df["전체프레임수"] / 2).astype(int).map({1: 0, 0: 1})
#         y_pred = (sub_df["정반응수"] >= threshold).astype(int).map({1: 0, 0: 1})
#
#
#         f1 = f1_score(y_true, y_pred, zero_division=0) * 100
#         acc = accuracy_score(y_true, y_pred) * 100
#         rec = recall_score(y_true, y_pred, zero_division=0) * 100
#         prec = precision_score(y_true, y_pred, zero_division=0) * 100
#
#         results.append({
#             "rec_group": rec_group,
#             "threshold": threshold,
#             "F1 Score": f1,
#             "Accuracy": acc,
#             "Recall": rec,
#             "Precision": prec
#         })
#
# # 결과 DataFrame
# result_df = pd.DataFrame(results)
# for row in result_df[result_df["Accuracy"] > 90].itertuples():
#     print(f"[DEBUG] {row.rec_group} | Threshold = {row.threshold} | Accuracy = {row._3:.2f}%")
#
# # 시각화
# metrics = ["F1 Score", "Accuracy", "Recall", "Precision"]
# for rec_group in ["rec_000", "rec_001"]:
#     act_group = rec_to_act[rec_group]
#     plt.figure(figsize=(10, 6))
#     subset = result_df[result_df["rec_group"] == rec_group]
#
#     # 선 그래프
#     for metric in metrics:
#         plt.plot(subset["threshold"], subset[metric], label=metric + " (%)")
#
#     # 강조 기준선 추가
#     for ht in highlight_thresholds:
#         plt.axvline(x=ht, color='gray', linestyle='--', linewidth=1)
#
#     plt.title(f"3DGazeNet {act_group} Result")
#     plt.xlabel("Threshold")
#     plt.ylabel("Score (%)")
#     plt.xticks(range(0, 151, 30))
#     plt.grid(True)
#     plt.legend(loc='lower right')  # 🔸 오른쪽 중앙 위치
#     plt.tight_layout()
#     plt.savefig(f"eval_metrics_{act_group}.png")
#     plt.show()

# import numpy as np
# import pandas as pd
# from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
# import matplotlib.pyplot as plt
#
# # 🔹 1. 예측 데이터 불러오기
# df_pred = pd.read_csv("C:/Users/user/Desktop/skh/ETRI/processed_videos/gaze_summary_metrics.csv")
# df_pred["정반응수"] = pd.to_numeric(df_pred["정반응수"], errors="coerce")
# df_pred["rec"] = df_pred["폴더명"].str.extract(r"(rec_0\d\d)")
# df_pred = df_pred.dropna(subset=["정반응수", "rec"])
#
# # GT 파일 (Sheet2) 불러오기 + 열 이름 직접 지정
# df_gt = pd.read_excel(
#     "C:/Users/user/Desktop/skh/3DGazenet/demo/AI모듈_ASD 태깅 정리.xlsx",
#     sheet_name="Sheet2",
#     header=0  # 첫 번째 줄이 헤더라면 이 줄은 생략 가능
# )
#
# # ID 컬럼 처리
# if "ID" not in df_gt.columns:
#     possible_id_col = df_gt.columns[0]
#     df_gt = df_gt.rename(columns={possible_id_col: "ID"})
# df_gt["ID"] = df_gt["ID"].astype(str).str.strip()
#
# # 하단 합계 제거
# df_gt = df_gt.iloc[:-2].reset_index(drop=True)
#
# # A2-1022-2 (회색으로 간주할 행 수동 제거)
# # ✅ '합동주시 반응2'에서만 NaN 처리
# df_gt.loc[df_gt["ID"].str.contains("A2-1022-2", na=False), "합동주시 반응2"] = np.nan
#
#
#
# # 숫자형 변환
# df_gt["합동주시 반응1"] = pd.to_numeric(df_gt["합동주시 반응1"], errors="coerce")
# df_gt["합동주시 반응2"] = pd.to_numeric(df_gt["합동주시 반응2"], errors="coerce")
#
# # GT 추출
# gt_000 = df_gt["합동주시 반응1"].dropna().astype(int).reset_index(drop=True)
# gt_001 = df_gt["합동주시 반응2"].dropna().astype(int).reset_index(drop=True)
#
# df_rec001 = df_pred[df_pred["rec"] == "rec_001"]
# print(df_rec001[["폴더명", "정반응수"]])
# print("🔍 예측 데이터 rec_001 개수:", len(df_rec001))
# print("✅ df_gt['합동주시 반응1'].notna():", df_gt["합동주시 반응1"].notna().sum())
# print("✅ df_gt['합동주시 반응2'].notna():", df_gt["합동주시 반응2"].notna().sum())
#
#
# # 🔹 3. 예측값과 GT 매칭 및 평가
# rec_map = {"rec_000": ("act1", gt_000), "rec_001": ("act2", gt_001)}
#
# for rec_code in ["rec_000", "rec_001"]:
#     act_name, y_true = rec_map[rec_code]
#     df_act = df_pred[df_pred["rec"] == rec_code].reset_index(drop=True)
#
#     # 🔸 ID 추출용 정규표현식 분기
#     if rec_code == "rec_000":
#         df_act["ID"] = df_act["폴더명"].str.extract(r"processed_(.*)_02_rec_000")
#         gt_ids = df_gt.loc[df_gt["합동주시 반응1"].notna(), "ID"].reset_index(drop=True)
#     else:
#         df_act["ID"] = df_act["폴더명"].str.extract(r"processed_(.*)_02_rec_001")
#         gt_ids = df_gt.loc[df_gt["합동주시 반응2"].notna(), "ID"].reset_index(drop=True)
#
#     pred_ids = df_act["ID"].reset_index(drop=True)
#     y_pred_source = df_act["정반응수"].values
#
#     if len(y_true) != len(y_pred_source):
#         print(f"\n⚠️ 길이 불일치 발생: {act_name}")
#         print(f"  - GT 개수:   {len(y_true)}")
#         print(f"  - 예측 개수: {len(y_pred_source)}")
#         print("  - 누락된 ID:", list(gt_ids[~gt_ids.isin(pred_ids)]))
#         continue
#
#     print(f"✅ 평가 진행: {act_name}")
#     # 평가 지표 계산 코드는 여기에 추가
#
#
#     # 평가 지표 저장
# import numpy as np
# import pandas as pd
# from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
# import matplotlib.pyplot as plt
#
# # 🔹 설정: 사용할 rec 코드 (rec_000 또는 rec_001)
# rec_code = "rec_001"  # ← 필요에 따라 rec_000으로 변경
#
# # 🔹 예측 파일 불러오기
# df_pred = pd.read_csv("C:/Users/user/Desktop/skh/ETRI/processed_videos/gaze_summary_metrics.csv")
# df_pred["정반응수"] = pd.to_numeric(df_pred["정반응수"], errors="coerce")
# df_pred["rec"] = df_pred["폴더명"].str.extract(r"(rec_0\d\d)")
# df_pred = df_pred.dropna(subset=["정반응수", "rec"])
# df_pred = df_pred[df_pred["rec"] == rec_code].reset_index(drop=True)
#
# # 🔹 GT 불러오기 (Sheet2)
# df_gt = pd.read_excel(
#     "C:/Users/user/Desktop/skh/3DGazenet/demo/AI모듈_ASD 태깅 정리.xlsx",
#     sheet_name="Sheet2",
#     header=0
# )
#
# if "ID" not in df_gt.columns:
#     df_gt = df_gt.rename(columns={df_gt.columns[0]: "ID"})
# df_gt["ID"] = df_gt["ID"].astype(str).str.strip()
# df_gt = df_gt.iloc[:-2].reset_index(drop=True)
#
# # A2-1022-2는 회색 셀 → 합동주시 반응2만 제거
# df_gt.loc[df_gt["ID"].str.contains("A2-1022-2", na=False), "합동주시 반응2"] = np.nan
#
# # 숫자형 변환
# df_gt["합동주시 반응1"] = pd.to_numeric(df_gt["합동주시 반응1"], errors="coerce")
# df_gt["합동주시 반응2"] = pd.to_numeric(df_gt["합동주시 반응2"], errors="coerce")
#
# # 🔹 GT 및 ID 추출
# if rec_code == "rec_000":
#     gt = df_gt["합동주시 반응1"].dropna().astype(int).reset_index(drop=True)
#     gt_ids = df_gt.loc[df_gt["합동주시 반응1"].notna(), "ID"].reset_index(drop=True)
#     df_pred["ID"] = df_pred["폴더명"].str.extract(r"processed_(.*)_02_rec_000")
# else:
#     gt = df_gt["합동주시 반응2"].dropna().astype(int).reset_index(drop=True)
#     gt_ids = df_gt.loc[df_gt["합동주시 반응2"].notna(), "ID"].reset_index(drop=True)
#     df_pred["ID"] = df_pred["폴더명"].str.extract(r"processed_(.*)_02_rec_001")
#
# # 🔹 ID 정렬 후 GT와 매칭
# df_pred = df_pred[df_pred["ID"].isin(gt_ids)].reset_index(drop=True)
# df_pred = df_pred.sort_values("ID").reset_index(drop=True)
# gt = gt[gt_ids.isin(df_pred["ID"])].reset_index(drop=True)
#
# # 🔹 평가할 threshold 목록 (30, 60, 90, 120)
# thresholds = [30, 60, 90, 120]
# metrics = {"Threshold": [], "F1": [], "Accuracy": [], "Recall": [], "Precision": []}
#
# for threshold in thresholds:
#     y_true = gt.values
#     # ✅ 반대로: threshold 이상이면 미반응(1), 미만이면 정반응(0)
#     y_pred = np.where(df_pred["정반응수"] >= threshold, 1, 0)
#
#     metrics["Threshold"].append(threshold)
#     metrics["F1"].append(f1_score(y_true, y_pred) * 100)
#     metrics["Accuracy"].append(accuracy_score(y_true, y_pred) * 100)
#     metrics["Recall"].append(recall_score(y_true, y_pred) * 100)
#     metrics["Precision"].append(precision_score(y_true, y_pred) * 100)
#
# # 🔹 DataFrame으로 변환 후 출력
# df_result = pd.DataFrame(metrics)
# print(df_result)
#
# # 🔹 시각화
# plt.figure(figsize=(10, 6))
# plt.plot(metrics["Threshold"], metrics["F1"], label="F1 Score (%)")
# plt.plot(metrics["Threshold"], metrics["Accuracy"], label="Accuracy (%)")
# plt.plot(metrics["Threshold"], metrics["Recall"], label="Recall (%)")
# plt.plot(metrics["Threshold"], metrics["Precision"], label="Precision (%)")
# plt.xlabel("Threshold")
# plt.ylabel("Score (%)")
# plt.title(f"3DGazeNet ({rec_code}) Result")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"./result_{rec_code}_threshold_plot.png")
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# 파일 경로 설정
tagging_path = "C:/Users/user/Desktop/skh/3DGazenet/demo/AI모듈_ASD 태깅 정리.xlsx"
metrics_path = "C:/Users/user/Desktop/skh/ETRI/processed_videos/gaze_summary_metrics.csv"

# 🔹 GT 불러오기 및 정제
df_tagging = pd.read_excel(tagging_path, sheet_name="Sheet2")
df_tagging.columns = df_tagging.columns.str.strip()
df_tagging = df_tagging.rename(columns={"부산치료실": "ID"})
df_tagging["ID"] = df_tagging["ID"].astype(str).str.strip()
df_tagging.loc[df_tagging["ID"].str.contains("A2-1022-2", na=False), "합동주시 반응2"] = pd.NA
df_tagging["합동주시 반응1"] = pd.to_numeric(df_tagging["합동주시 반응1"], errors="coerce")
df_tagging["합동주시 반응2"] = pd.to_numeric(df_tagging["합동주시 반응2"], errors="coerce")

# 🔹 예측 파일 불러오기 및 정제
df_metrics = pd.read_csv(metrics_path)
df_metrics["정반응수"] = pd.to_numeric(df_metrics["정반응수"], errors="coerce")
df_metrics["전체프레임수"] = pd.to_numeric(df_metrics["전체프레임수"], errors="coerce")
df_metrics["rec"] = df_metrics["폴더명"].str.extract(r"(rec_0\d\d)")
df_metrics["ID"] = df_metrics["폴더명"].str.extract(r"processed_(.*)_02_")
df_metrics = df_metrics.dropna(subset=["정반응수", "전체프레임수", "rec", "ID"])

# 🔹 병합
df_merged = df_metrics.merge(df_tagging[["ID", "합동주시 반응1", "합동주시 반응2"]], on="ID", how="inner")

# 🔹 threshold 기반 평가지표 계산 (1~150 프레임 단위)
thresholds = list(range(1, 151))
metrics_all = []

for rec_group in ["rec_000", "rec_001"]:
    gt_col = "합동주시 반응1" if rec_group == "rec_000" else "합동주시 반응2"
    df_sub = df_merged[df_merged["rec"] == rec_group].dropna(subset=[gt_col])
    y_true = df_sub[gt_col].astype(int).values
    for t in thresholds:
        y_pred = (df_sub["정반응수"] < t).astype(int).values  # threshold보다 작으면 미반응(1)
        metrics_all.append({
            "rec_group": rec_group,
            "threshold": t,
            "F1 Score": f1_score(y_true, y_pred) * 100,
            "Accuracy": accuracy_score(y_true, y_pred) * 100,
            "Recall": recall_score(y_true, y_pred) * 100,
            "Precision": precision_score(y_true, y_pred) * 100
        })

# 🔹 DataFrame으로 저장
df_metrics_all = pd.DataFrame(metrics_all)

# 🔹 시각화 함수 정의
def plot_metrics(df, rec_group, save_path):
    plt.figure(figsize=(10, 6))
    subset = df[df["rec_group"] == rec_group]
    for metric in ["F1 Score", "Accuracy", "Recall", "Precision"]:
        plt.plot(subset["threshold"], subset[metric], label=metric)
    for x in [30, 60, 90, 120]:
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Evaluation Metrics by Threshold ({rec_group})")
    plt.xlabel("Threshold (Frames)")
    plt.ylabel("Score (%)")
    plt.xticks(range(0, 151, 30))
    plt.ylim(0, 110)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 🔹 그래프 저장
plot_metrics(df_metrics_all, "rec_000", "rec_000_metrics_fullrange_plot.png")
plot_metrics(df_metrics_all, "rec_001", "rec_001_metrics_fullrange_plot.png")

df_metrics_selected = df_metrics_all[df_metrics_all["threshold"].isin([30, 60, 90, 120])]
print(df_metrics_selected.round(2))