from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class KiTS23Dataset(Dataset):
    """
    Minimal 3D KiTS23 dataset loader for MedNeXt.

    Labels:
        0 = background
        1 = kidney
        2 = tumor
        3 = cyst

    The dataset extracts 3D patches from the original KiTS23
    NIfTI volumes. No 2D slices are used.
    """

    def __init__(
        self,
        data_root,
        patch_size=(32, 128, 128),
        cases=None,
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size

        if cases is None:
            self.cases = sorted(
                [
                    p for p in self.data_root.iterdir()
                    if p.is_dir() and p.name.startswith("case_")
                ]
            )
        else:
            self.cases = [
                self.data_root / case
                for case in cases
            ]

        if len(self.cases) == 0:
            raise RuntimeError(
                f"No KiTS23 cases found in {self.data_root}"
            )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case_dir = self.cases[index]

        image_path = case_dir / "imaging.nii.gz"
        label_path = case_dir / "segmentation.nii.gz"

        # Load original 3D NIfTI volumes
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)

        image = np.asarray(image_nii.dataobj, dtype=np.float32)
        label = np.asarray(label_nii.dataobj, dtype=np.int64)

        # Basic CT normalization.
        # We deliberately keep this simple for the first pipeline test.
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000

        # Extract a patch.
        image_patch, label_patch = self._extract_patch(
            image,
            label,
        )

        # Add channel dimension:
        # [D, H, W] -> [1, D, H, W]
        image_patch = torch.from_numpy(
            image_patch.copy()
        ).unsqueeze(0)

        label_patch = torch.from_numpy(
            label_patch.copy()
        )

        return {
            "image": image_patch,
            "label": label_patch,
            "case": case_dir.name,
        }

    def _extract_patch(self, image, label):
        pd, ph, pw = self.patch_size

        d, h, w = image.shape

        if d < pd or h < ph or w < pw:
            raise ValueError(
                f"Patch {self.patch_size} is larger than "
                f"volume {image.shape}"
            )

        # For this first test, take a deterministic central crop.
        d_start = (d - pd) // 2
        h_start = (h - ph) // 2
        w_start = (w - pw) // 2

        image_patch = image[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw,
        ]

        label_patch = label[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw,
        ]

        return image_patch, label_patch