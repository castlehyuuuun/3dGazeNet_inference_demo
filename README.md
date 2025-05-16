# 3dGazeNet_inference_demo

To run inference on a set of images follow the steps below. A set of example images are given in the data/example_images directory.

For video, 
$ python inference_video.py --cfg configs/infer_res34_x128_all_vfhq_vert.yaml 
                            --video_path data/test_videos/ms_30s.mp4 
                            --smooth_predictions
utils 파일은 utils directory 안의 utils 파일을 의미함
위 파일 예시로 python '경로상의 inference_video.py' --cfg infer_res18_x128_all_vfhq_vert.yaml --video_path '경로상의 데이터셋' --smooth_predictions
