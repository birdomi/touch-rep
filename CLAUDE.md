# touch-rep 프로젝트 가이드

## 프로젝트 개요

촉각(tactile) 센서 데이터를 위한 Self-Supervised Learning(SSL) 프레임워크.
XELA, BrainCo, DIGIT360, GelSight 등 다양한 촉각 센서를 지원하며,
DINOv2, BYOL, MAE 알고리즘으로 표현 학습 후 downstream task에 적용.

## 실행 환경

```bash
conda activate tactile_ssl
```

- Python 3.9, PyTorch 2.0, CUDA 11.8
- Lightning Fabric(분산학습), Hydra(config), wandb(실험 추적)
- xformers (efficient attention)

## 주요 디렉토리

```
touch-rep/
├── tactile_ssl/
│   ├── algorithm/          # SSL 알고리즘 (DINOv2, BYOL, MAE)
│   ├── model/              # Transformer 아키텍처
│   │   ├── brainco_transformer.py   # BrainCo 전용 (NullAwarePatchEmbed 포함)
│   │   ├── xela_transformer.py
│   │   ├── signal_transformer.py    # 베이스 클래스
│   │   └── layers/                  # attention, block, mlp 등
│   ├── data/               # 데이터셋 클래스
│   │   ├── brainco_grasp_dataset.py      # Grasp detection/prediction 데이터
│   │   ├── brainco_grasp_multimodal_dataset.py  # 촉각+RGB 멀티모달
│   │   └── brainco_tactile.py            # BrainCo SSL 데이터
│   ├── downstream_task/    # 파인튜닝 모듈
│   │   ├── sl_module.py               # 베이스 클래스
│   │   ├── brainco_grasp_sl.py        # Grasp 분류 (tactile 전용)
│   │   ├── brainco_grasp_fusion_sl.py # Grasp 분류 (tactile + vision)
│   │   └── multi_sensor_grasp_sl.py   # 멀티센서
│   ├── loss/               # DINO loss, iBOT, KoLeo
│   ├── trainer/            # Lightning Fabric 기반 Trainer
│   └── utils/              # logging, masking, EMA 등
├── config/
│   ├── default.yaml        # SSL 사전학습 기본 config
│   ├── default_task.yaml   # downstream task 기본 config
│   ├── algorithm/          # 알고리즘별 하이퍼파라미터
│   ├── data/               # 데이터셋 config
│   ├── task/               # downstream task config
│   ├── experiment/         # 실험 조합 config (override)
│   └── paths/default.yaml  # 데이터/체크포인트 경로
├── train.py                # SSL 사전학습 진입점
├── train_task.py           # Downstream task 학습 (XELA/D360)
└── train_task_brainco.py   # BrainCo grasp task 학습
```

## 학습 실행 방법

```bash
# SSL 사전학습
python train.py +experiment=brainco/dinov2.yaml

# BrainCo grasp prediction (단일 fold)
python train_task_brainco.py +experiment=brainco/task/grasp_prediction/dinov2.yaml

# K-fold 전체 실행
python train_task_brainco.py +experiment=brainco/task/grasp_prediction/dinov2.yaml --all_split

# fold 수 지정
python train_task_brainco.py +experiment=... --all_split --num_folds 5
```

## 데이터 경로 구조

```
dataset/brainco/
├── ssl/                        # SSL 사전학습 데이터
├── downstream/
│   ├── grasp_detection/        # 그립 성공/실패 감지
│   │   ├── grasp_success/episode_XXXX/
│   │   └── grasp_fail/episode_XXXX/
│   └── grasp_prediction/       # 그립 예측 (동일 구조)
│       └── {grasp_success|grasp_fail}/episode_XXXX/
│           ├── data.json        # 관절각도, 촉각 데이터 (61 frames 기준)
│           ├── tactiles/        # numpy 배열 파일들
│           └── colors/          # RGB 이미지 프레임들
└── urdf/                       # 로봇 URDF 파일
```

## 핵심 데이터 형식

### BrainCo 촉각 센서
- **센서 배치**: 10개 (왼손 5 + 오른손 5, 각 손가락 1개)
- **채널**: 4개
  - ch0: normal force (범위 0~25000, 정상적인 숫자)
  - ch1: tangential force 
  - ch2: tangential direction — **거의 모든 프레임에서 65535(invalid) → -1로 대체됨** (force가 없을 때는 65535)
  - ch3: proximity (범위 0~451,000 — 매우 큰 스케일 주의!)
- **입력 shape**: `(B, W, N=10, C=4)`  — W=window size(time frames)

### 촉각 데이터 주의사항
- **ch2는 대부분 invalid(-1)**: normalization 전에 null 처리 필수
- **ch3(proximity) 스케일 불균형**: ch0 대비 ~20,000배 — 반드시 normalization 필요
- `NullAwarePatchEmbed`가 -1값 처리 담당 (model/brainco_transformer.py)

## BrainCo 모델 아키텍처

```
입력 (B, T=1, N=10, C=4)
  ↓ NullAwarePatchEmbed  (null 채널 감지 → zero-fill → linear + null_embed)
  ↓ PositionEmbed        (fingertip 6D pose → embed_dim)
  ↓ sensor_block         (4 layers, spatial attention across sensors)
  ↓ transformer blocks   (8 layers total, pre_fusion_block_idx=4에서 pos 합산)
  ↓ LayerNorm
출력 (B, num_register_tokens + N, embed_dim=192)
```

### 모델 설정 (brainco_tiny)
```yaml
embed_dim: 192
depth: 8
num_heads: 3
num_register_tokens: 1
sequence_length: 1      # 현재: 단일 시점 처리
time_chunk_size: 1
in_dim: 10              # 센서 개수
in_chans: 4             # 채널 수
```

## Grasp Task 데이터 흐름

```
BraincoGraspDetectionDataset
  window_time=0.1s, window_overlap=0.5 → 10 frames/window
  61 frames/episode → ~10 windows/episode

batch["sensor"]       : (B, W=10, N=10, C=4)  ← W: window 내 time frame 수
batch["sensor_poses"] : (B, W=10, N=10, 6)    ← fingertip 6D pose
batch["label"]        : (B,)                   ← 0=fail, 1=success

encode():
  sensor.view(B*W, 1, N, C) → BraincoTransformer (각 frame 독립 처리)
  → AttentivePooler → (B, W, embed_dim=192)
  → BraincoGraspProbe (causal temporal attention)
  → (B, 2)
```

## K-Fold 분할 방식

- **연속 블록 분할** (round-robin 아님)
- fold k = episode를 정렬 후 k번째 연속 구간을 val로 사용
- 예: num_folds=5, fold=0 → episode 0~19%가 val
- `get_dataloader_brainco_grasp()` in `train_task_brainco.py`

## Config 시스템 (Hydra)

```yaml
# experiment config 구조 예시
defaults:
  - override /data: brainco_grasp_prediction
  - override /task: brainco_grasp_prediction
  - _self_

task:
  checkpoint_encoder: ${paths.encoder_checkpoint_root}/dinov2_brainco_tiny/epoch-0300.ckpt
  train_encoder: true   # encoder 파인튜닝 여부
```

- `${paths.data_root}` → `config/paths/default.yaml`에 정의
- custom resolver: `${int_multiply(a,b)}`, `${capitalize(str)}`

## 코딩 시 주의사항

### 1. 텐서 shape 컨벤션
```python
# SSL 사전학습 입력
(B, T, N, C)  # batch, time, num_sensors, channels

# Grasp task 입력
(B, W, N, C)  # batch, window_frames, num_sensors, channels

# 인코더 입력 (각 window 독립 처리 시)
sensor.view(B*W, 1, N, C)
```

### 2. Null 채널 처리
```python
# ch2가 -1인 경우 NullAwarePatchEmbed가 자동 처리
# BUT 직접 normalization할 때는 반드시 null 제거 후 진행:
null_mask = (x[..., :4] < 0)
x_clean = x.clone()
x_clean[null_mask] = 0.0
x_norm = normalize(x_clean)
```

### 3. Pretrained 체크포인트 로딩
```python
# SLModule이 checkpoint_encoder를 자동으로 로드함
# 단, in_chans나 sequence_length가 다르면 key mismatch 발생
# → strict=False 또는 weight copy 필요
```

### 4. wandb 설정
- `wandb/settings` 파일에 `mode = online` 설정 시 실제 wandb 서버로 전송
- 오프라인 테스트: `mode = disabled`
- API key: `~/.netrc`에 저장됨

### 5. Hydra config 파라미터 추가
- task/data config에 없는 키를 experiment에서 쓰면 `InterpolationKeyError` 발생
- 최상위 config(`default_task.yaml`)에 기본값 추가 필요

### 6. 새 downstream task 추가 시
```
1. tactile_ssl/data/brainco_new_dataset.py       — 데이터셋 클래스
2. tactile_ssl/downstream_task/new_sl.py         — SLModule 상속
3. tactile_ssl/downstream_task/__init__.py       — import 추가
4. config/data/brainco_new.yaml                  — 데이터 config
5. config/task/brainco_new_task.yaml             — task config
6. config/experiment/brainco/task/new/exp.yaml   — 실험 config
7. train_task_brainco.py get_dataloaders()에 sensor 분기 추가
```

### 7. Episode-level 데이터 구조
```python
dataset.episode_data[i] = {
    "path": str,           # episode 디렉토리 경로
    "tactile_array": np.ndarray,  # (N_frames, 10, 4)
    "fingertip_6d": np.ndarray,   # (N_frames, 10, 6)
    "window_starts": np.ndarray,  # 각 window의 시작 frame 인덱스
    "label": int,          # 0=fail, 1=success
}
# windows: episode_data를 window 단위로 flatten한 것
dataset.windows[i] = {
    "sensor": Tensor,             # (W, 10, 4)
    "sensor_poses": Tensor,       # (W, 10, 6)
    "label": Tensor,
    "episode_path": str,
    "window_start_frame": int,
}
```

## 실험 결과 (참고)

| 모델 | Task | Mean Acc | Mean F1 |
|------|------|----------|---------|
| ResNet18 (vision only) | grasp_prediction | ~0.988 | ~0.988 |
| BraincoTransformer (tactile) | grasp_prediction | 낮음 (분석 중) |
| Fusion (tactile + vision) | grasp_prediction | 실험 중 |

**Tactile 성능 저하 주요 원인:**
1. ch3 (proximity) 스케일 불균형 (normalization 미적용)
2. ch2 (depth) invalid 값 처리 부재 → NullAwarePatchEmbed로 개선됨
3. sequence_length=1 → window 내 temporal context 부재

## 주요 체크포인트 위치

```
checkpoints/
└── dinov2_brainco_tiny/
    └── epoch-0300.ckpt    # BrainCo SSL pretrained (brainco_tiny 아키텍처)
```
