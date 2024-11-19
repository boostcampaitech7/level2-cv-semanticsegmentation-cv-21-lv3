# debug_dataset.py
import os
from torch.utils.data import DataLoader
import numpy as np
import json
import cv2

data_root = "/home/minipin/level2_cv_semanticsegmentation-cv-03/data/train"

dataset = XRayboneDataset(data_root=data_root, split="train")
dataloader = DataLoader(dataset, batch_size=4, num_workers=8, shuffle=False)

# 1. 데이터셋 경로 확인
data_list = dataset.load_data_list()
for data in data_list:
    print(f"Image path: {data['img_path']}")
    print(f"Segmentation map path: {data['seg_map_path']}")
    if not os.path.exists(data['img_path']):
        print(f"Missing image file: {data['img_path']}")
    if not os.path.exists(data['seg_map_path']):
        print(f"Missing seg map file: {data['seg_map_path']}")

# 2. 데이터셋에서 개별 아이템 확인
for idx in range(5):  # 첫 5개 데이터 확인
    data = dataset[idx]
    print(f"Index {idx} data: {data}")

# 3. Dataloader 확인
for batch in dataloader:
    print(f"Batch: {batch}")
    break

PALETTE = {
    "Radius": (220, 20, 60),
    "finger-1": (119, 11, 32),
    "finger-2": (0, 0, 142),
    "finger-3": (0, 0, 230),
    "finger-4": (106, 0, 228),
    "finger-5": (0, 60, 100),
    "finger-6": (0, 80, 100),
    "finger-7": (0, 0, 70),
    "finger-8": (0, 0, 192),
    "finger-9": (250, 170, 30),
    "finger-10": (100, 170, 30),
    "finger-11": (220, 220, 0),
    "finger-12": (175, 116, 175),
    "finger-13": (250, 0, 30),
    "finger-14": (165, 42, 42),
    "finger-15": (255, 77, 255),
    "finger-16": (0, 226, 252),
    "finger-17": (182, 182, 255),
    "finger-18": (0, 82, 0),
    "finger-19": (120, 166, 157),
    "Trapezoid": (110, 76, 0),
    "Scaphoid": (174, 57, 255),
    "Trapezium": (199, 100, 0),
    "Lunate": (72, 0, 118),
    "Triquetrum": (255, 179, 240),
    "Hamate": (0, 125, 92),
    "Capitate": (209, 0, 151),
    "Ulna": (188, 208, 182),
    "Pisiform": (0, 220, 176)
}


class XRayDataset(BaseSegDataset):
    def __init__(self, data_root, split, **kwargs):
        self.data_root = data_root
        self.split = split
        self.seg_map_dir = os.path.join(data_root, "seg_maps")  # Segmentation maps 저장 경로
        super().__init__(**kwargs, data_root=data_root)


    def load_data_list(self):
        data_list = []
        json_root = os.path.join(self.data_root, "outputs_json")
        img_root = os.path.join(self.data_root, "DCM")

        for id_folder in os.listdir(json_root):
            id_folder_path = os.path.join(json_root, id_folder)
            if not os.path.isdir(id_folder_path):
                continue
            
            for json_file in sorted(os.listdir(id_folder_path)):
                if json_file.endswith(".json"):
                    json_path = os.path.join(id_folder_path, json_file)
                    img_path = os.path.join(img_root, id_folder, json_file.replace(".json", ".png"))

                    if not os.path.exists(img_path):
                        print(f"Image not found for {json_path}")
                        continue

                    # Determine segmentation map path
                    seg_map_name = f"{id_folder}_{json_file.replace('.json', '.png')}"
                    seg_map_path = os.path.join(self.seg_map_dir, seg_map_name)
                    
                    # Check if segmentation map already exists
                    if not os.path.exists(self.seg_map_dir):
                        os.mkdir(self.seg_map_dir)
                        self.create_seg_map(json_path, seg_map_path)
                    # Append to data list
                    data_info = {
                        "img_path": img_path,
                        "seg_map_path": seg_map_path,
                    }
                    data_list.append(data_info)

        return data_list

    def create_seg_map(self, json_path, seg_map_path):
        # 이미지 크기 (Assume fixed size, e.g., 2048x2048)
        img_size = (2048, 2048)  # 단일 채널(Grayscale)
        seg_map = np.zeros(img_size, dtype=np.uint8)

        # Parse JSON file
        with open(json_path, "r", encoding='utf-16') as f:
            annotation_data = json.load(f)

        # Draw polygons
        for annotation in annotation_data["annotations"]:
            label = annotation["label"]
            class_index = self.label_to_index(label)  # 클래스 인덱스 가져오기
            points = np.array(annotation["points"], dtype=np.int32)
            cv2.fillPoly(seg_map, [points], class_index)

        # Save segmentation map as .png (Grayscale)
        cv2.imwrite(seg_map_path, seg_map)

    def label_to_index(self, label):
        # Label-to-index mapping (Example: customize as needed)
        label_mapping = {k: i + 1 for i, k in enumerate(PALETTE.keys())}  # 1부터 시작
        return label_mapping.get(label, 0)  # Default to 0 (background)
    


class BaseSegDataset(BaseDataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of BaseSegDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as
            specify classes to load. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img_path=None, seg_map_path=None).
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=True``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    METAINFO: dict = dict()

    def __init__(
        self,
        ann_file: str = "",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img_path="", seg_map_path=""),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        ignore_index: int = 255,
        reduce_zero_label: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get("classes", None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(label_map=self.label_map, reduce_zero_label=self.reduce_zero_label)
        )

        # Update palette based on label map or generate palette
        # if it is not defined
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert (
                self._metainfo.get("classes") is not None
            ), "dataset metainfo `classes` should be specified when testing"

    @classmethod
    def get_label_map(cls, new_classes: Optional[Sequence] = None) -> Union[Dict, None]:
        """Require label mapping.

        The ``label_map`` is a dictionary, its keys are the old label ids and
        its values are the new label ids, and is used for changing pixel
        labels in load_annotations. If and only if old classes in cls.METAINFO
        is not equal to new classes in self._metainfo and nether of them is not
        None, `label_map` is not None.

        Args:
            new_classes (list, tuple, optional): The new classes name from
                metainfo. Default to None.


        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        old_classes = cls.METAINFO.get("classes", None)
        if (
            new_classes is not None
            and old_classes is not None
            and list(new_classes) != list(old_classes)
        ):
            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO["classes"]):
                raise ValueError(
                    f"new classes {new_classes} is not a "
                    f"subset of classes {old_classes} in METAINFO."
                )
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        If length of palette is equal to classes, just return the palette.
        If palette is not defined, it will randomly generate a palette.
        If classes is updated by customer, it will return the subset of
        palette.

        Returns:
            Sequence: Palette for current dataset.
        """
        palette = self._metainfo.get("palette", [])
        classes = self._metainfo.get("classes", [])
        # palette does match classes
        if len(palette) == len(classes):
            return palette

        if len(palette) == 0:
            # Get random state before set seed, and restore
            # random state later.
            # It will prevent loss of randomness, as the palette
            # may be different in each iteration if not specified.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            new_palette = np.random.randint(0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            # return subset of palette
            for old_id, new_id in sorted(self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError(
                "palette does not match classes " f"as metainfo is {self._metainfo}."
            )
        return new_palette

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args
            )
            for line in lines:
                img_name = line.strip()
                data_info = dict(img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["seg_fields"] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args,
            ):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["seg_fields"] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x["img_path"])
        return data_list
