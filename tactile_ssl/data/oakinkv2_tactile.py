from tactile_ssl.data.gigahands_tactile import XYZCHandDataset


class OakInkV2TactileDataset(XYZCHandDataset):
    """OakInk-v2 hand keypoint pretraining dataset.

    데이터 경로: pretraining_dataset/OakInkv2/
    파일 포맷:  scene_XX__AXXX++seq__<hash>__<timestamp>.pkl

    scenes 파라미터로 특정 scene만 로드:
        scenes=['scene_01']  →  'scene_01__'로 시작하는 파일만 사용
    """

    def __init__(self, data_root: str = "pretraining_dataset/OakInkv2", **kwargs):
        super().__init__(data_root=data_root, **kwargs)
