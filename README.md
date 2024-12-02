 # Boostcamp AI Tech 7 CV 21
 
## Hand X-ray Bone semantic Segmentation
### 2024.11.13 10:00 ~ 2024.11.28 19:00

![image](https://github.com/user-attachments/assets/2046c625-00b8-48e6-b63a-505076dccddd)
## Description
 뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치는 신체 조직이다. 이에 따라 정확한 Bone Segmentation은 의료 진단 및 치료 계획을 개발하는 데 필수적인 역할을 수행하며, 이에 따라 bone segmentation은 딥러닝 기반 인공지능 분야의 중요한 응용 분야로써 다양한 연구가 이뤄지고 있다. 그럼에도 정확한 bone segmentation은 여전히 많은    어려움을 수반한다. 이 문제를 해결하기 위해 본 프로젝트에서는 hand bone X-ray 이미지에서 29가지의 뼈 클래스에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀에 해당 클래스를 할당하는 Semantic Segmentation 모델을 개발하고자 한다.

## 평가 지표

이번 대회의 평가 방식은 Dice coefficient 기반의 Mean Dice로 평가한다. 
<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/f629bbd2-baa0-4e28-8c68-ab01d437c215" alt="Dice Coefficient" style="width: 30%; height: auto;">
</div>

Dice coefficient는 Segmentation에서 사용하는 대표적인 성능 측정 방법이며, Ground Truth 픽셀 클래스 집합과 Prediction 픽셀 클래스 집합 간 일치도를 기준으로 교집합의 두 배 값을 합집합 값으로 나눈 값으로 계산한다. Mean Dice는 클래스별 Dice coefficient의 평균으로 계산된다

## Result
![result](https://github.com/user-attachments/assets/76105558-5699-4d1a-b305-e68e6245a60a)

최종 dice 0.9750로 리더보드 순위 13등 달성


## Contributor
| <img src="https://github.com/user-attachments/assets/a669d334-7820-4e28-8a05-5a9d745ddc42" alt="박동준" style="width:100px; height:100px;"> | <a href="https://github.com/Ahn-latte"><img src="https://avatars.githubusercontent.com/Ahn-latte" alt="안주형" style="width:100px; height:100px;"></a> | <a href="https://github.com/minseokheo"><img src="https://avatars.githubusercontent.com/minseokheo" alt="허민석" style="width:100px; height:100px;"></a> | <a href="https://github.com/leedoyoung6"><img src="https://avatars.githubusercontent.com/leedoyoung6" alt="이도영" style="width:100px; height:100px;"></a> | <a href="https://github.com/MinSeok1204"><img src="https://avatars.githubusercontent.com/MinSeok1204" alt="최민석" style="width:100px; height:100px;"></a> | <a href="https://github.com/airacle100"><img src="https://avatars.githubusercontent.com/airacle100" alt="윤정우" style="width:100px; height:100px;"></a> |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| [박동준](https://github.com/Poodlee)                                               | [안주형](https://github.com/Ahn-latte)                                                   | [허민석](https://github.com/minseokheo)                                              | [이도영](https://github.com/leedoyoung6)                                                  | [최민석](https://github.com/MinSeok1204)             | [윤정우](https://github.com/airacle100)             |


## Project Structure

```
├── README.md                           
├── config/
│   └── config.json                     # 모델 및 데이터 설정 관련 JSON 파일
├── mmsegmentation/                     # MMSegmentation 관련 코드 및 구성
│   ├── test.py                         # 학습된 MMSeg 모델을 테스트하는 스크립트
│   ├── train.py                        # MMSeg 모델 학습 스크립트
│   ├── mmseg/
│   │   ├── datasets/                   # 데이터셋 정의 관련 스크립트
│   │   │   ├── xray_bone.py            # X-ray 데이터셋 커스텀 데이터 정의
│   │   │   ├── xray_bone2.py           # X-ray 데이터셋의 다른 커스텀 정의
│   │   ├── evaluation/                 # 모델 평가 관련 코드
│   │   │   ├── metrics/                # 평가 지표 관련 코드
│   │   │   │   ├── dice_xraybone.py    # Dice 계수를 사용한 X-ray 데이터셋 평가 코드
│   │   ├── models/                     # 모델 정의 관련 코드
│   │   │   ├── segmentors/             # 세그멘테이션 모델 정의
│   │   │   │   ├── xray_bone.py        # X-ray 데이터셋을 위한 세그멘테이션 모델 정의
│   ├── segformer/                      # SegFormer 모델 관련 코드
│   │   ├── config.py                   # SegFormer 모델 설정 파일
│   │   ├── default_runtime.py          # 기본 런타임 설정 파일
│   │   ├── schedule_160k.py            # 학습 스케줄 설정 파일
│   │   ├── segformer_mit-b0.py         # SegFormer MIT-B0 모델 설정 파일
│   │   ├── segformer_mit-b0_8xb2-160k_ade20k-512x512.py  # SegFormer 학습 스크립트 설정
│   │   ├── xraydata2.py                # SegFormer용 데이터 로더
├── models/                             # 모델 관련 코드
│   ├── attn_unet.py                    # Attention U-Net 모델 구현
│   ├── model.py                        # 기타 커스텀 모델 정의
│   ├── __init__.py                     # 모델 관련 초기화 파일
├── trainer/                            # 모델 학습 및 관리 코드
│   ├── trainer.py                      # 모델 학습 관리 스크립트
│   ├── __init__.py                     # Trainer 관련 초기화 파일
├── utils/                              # 유틸리티 코드
│   ├── dataset.py                      # 데이터셋 처리 관련 유틸리티
│   ├── loss.py                         # 손실 함수 정의
│   ├── optimizer.py                    # 옵티마이저 정의
│   ├── scheduler.py                    # 학습 스케줄러 정의
│   ├── soft_voting.py                  # 앙상블을 위한 소프트 보팅 구현
│   ├── transforms.py                   # 데이터 변환 및 전처리 정의
│   ├── util.py                         # 기타 유틸리티 함수
│   ├── __init__.py                     # Utils 초기화 파일
├── requirements.txt                    # 프로젝트에 필요한 Python 패키지 리스트
├── run_train.sh                        # 학습 실행을 위한 셸 스크립트
├── test.py                             
├── train.py                            

```

## Usage

### Data Preparation
데이터셋을 `data/` 디렉토리에 준비한다.


### Training
   ```
   # config.json 수정 후
   python train.py
   # shell script
   ./run_train.sh
   ```

### Inference
   ```
   python test.py
   ```


## Requirements
```
torch==2.4.1
numpy==1.24.4
opencv-python==4.10.0.84
albumentations==1.4.18
pandas==2.2.3
scikit-learn==1.5.2
albumentations==1.4.18
matplotlib==3.9.2
segmentation-models-pytorch==0.3.0
tqdm==4.65.2
wandb==0.18.7
lion-pytorch==0.2.2
```
## Citation

```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
