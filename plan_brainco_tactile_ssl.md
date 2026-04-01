# BraincoSSLDataset 재작성 계획

## 현황

| 파일 | 상태 |
|------|------|
| `tactile_ssl/data/brainco_tactile_prev.py` | 구 버전 (단일 에피소드, FK 캐싱 없음) |
| `tactile_ssl/data/brainco_tactile.py` | **비어 있음** — 신규 작성 대상 |
| `config/data/brainco.yaml` | `_target_: tactile_ssl.data.brainco_tactile.BraincoSSLDataset` 가리킴 |

`train.py::get_dataloaders_brainco_based()`는 에피소드 단위로
`BraincoSSLDataset(config=..., data_path="{root}/{object}/episode_XXXX", ...)` 를 호출한다.
즉 **클래스 인터페이스는 prev와 동일하게 단일 에피소드** 로딩이 맞음.

---

## prev 버전의 문제점

1. **FK 매 실행 재계산** — `__init__`에서 모든 프레임에 대해 `compute_combined_fk()` 호출.
   에피소드당 수백 프레임 × 오브젝트 수 × 에피소드 수 → 학습 시작 시간 수 분.

4. **`joint_angles` 반환 불필요** — `SensorIdDatasetWrapper`가 `{sensor, sensor_poses, object_classification, sensor_id}` 만 보존.
   joint_angles 계산·반환은 낭비.

5. **65535 처리 위치** — `__init__` 에서 일괄 처리(OK)하지만, 이후 normalization 시
   null(-1) 마스크를 별도로 처리하지 않으면 통계가 오염됨.

---

## 신규 클래스 설계

### 인터페이스 (변경 없음)

```python
BraincoSSLDataset(
    config: DictConfig,      # window_time, window_overlap, interpolating_freq,
                             # subtract_baseline, smooth_data, bias_noise_std, bias_range
    data_path: str,          # {root}/{object}/episode_XXXX  (단일 에피소드 디렉토리)
    brainco_urdf_path: str,  # URDF 루트
    object_class: int | None,
    load_images: bool = False,
    joint_poses: bool = False, # False일 경우, sensor_poses가 finger tip only. True일 경우, 21 joint 들에 대한 pose
)
```

### `__getitem__` 출력 (변경 새로 작성)

```python
{
    "sensor":       Tensor(W, 10, 4),   # float32, normalized, ch2 null→0 (기존과 같음)
    "sensor_poses": Tensor(W, 10, 3),   # 3D fingertip pose or Joint pose (자기 손목에 대한 상대좌표)
    "wrist_poses": Tensor(W, 2, 9), # 3D translation + 6D rotation relative to virtual root (Virtual Root를 생성해야함.)
    "object_classification": Tensor,    # optional
}
```


## 파일 구조 (최종)

```
tactile_ssl/data/brainco_tactile.py
├── build_combined_chain()      ← prev에서 그대로 복사
├── compute_combined_fk()       ← prev에서 그대로 복사
└── BraincoSSLDataset
    ├── __init__()              ← joint_angles 제거
    ├── read_fingertip_sample() ← 동일
    ├── read_wrist_sample()     ← 동일
    └── __getitem__()           
```

---

## Config 변경 불필요

`config/data/brainco.yaml` 은 이미 `subtract_baseline: False`, `smooth_data: False`
키를 포함하고 있으므로 수정 불필요.

---

## 검증 방법

```bash
# 단일 에피소드 로드 테스트 (캐시 생성 확인)
python -c "
from omegaconf import OmegaConf
from tactile_ssl.data.brainco_tactile import BraincoSSLDataset
cfg = OmegaConf.create({'window_time':0.1,'window_overlap':0.5,
    'interpolating_freq':100,'subtract_baseline':False,
    'smooth_data':False,'bias_noise_std':0,'bias_range':0})
ds = BraincoSSLDataset(cfg, 'dataset/brainco/pretraining/basket/episode_0000',
                       'dataset/brainco/urdf')
print(ds[0]['sensor'].shape)        # (10, 10, 4)
print(ds[0]['sensor_poses'].shape)  # (10, 10, 3)
"

# 캐시 파일 확인
ls dataset/brainco/pretraining/basket/episode_0000/*cache.npy

# 전체 학습 실행
python train.py +experiment=brainco/dinov2.yaml
```
