from pathlib import Path

from matplotlib import image
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
        sampling_mode="foreground",
    ):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.sampling_mode = sampling_mode

        valid_modes = {
            "foreground",
            "tumor",
            "cyst",
        }

        if self.sampling_mode not in valid_modes:
            raise ValueError(
                f"Invalid sampling_mode: {self.sampling_mode}. "
                f"Choose from {valid_modes}."
            )

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

        while True:

            image_path = case_dir / "imaging.nii.gz"
            label_path = case_dir / "segmentation.nii.gz"

            # Load original 3D NIfTI volumes
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)

            image = np.asarray(
                image_nii.dataobj,
                dtype=np.float32
            )

            label = np.asarray(
                label_nii.dataobj,
                dtype=np.int64
            )

            # For cyst sampling, make sure this case
            # actually contains cyst voxels.
            if self.sampling_mode == "cyst":
                if not np.any(label == 3):

                    # Select another random case.
                    random_index = np.random.randint(
                        len(self.cases)
                    )

                    case_dir = self.cases[random_index]

                    continue

            break

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

        if h < ph or w < pw:
            raise ValueError(
                f"Patch {self.patch_size} is larger than volume "
                f"{image.shape}"
            )

        if d < pd:
            pad_d = pd - d

            image = np.pad(
                image,
                ((0, pad_d), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            label = np.pad(
                label,
                ((0, pad_d), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            d, h, w = image.shape

        # -----------------------------------------------------
        # Select target voxels
        # -----------------------------------------------------

        if self.sampling_mode == "foreground":
            target_voxels = np.argwhere(label > 0)

        elif self.sampling_mode == "tumor":
            target_voxels = np.argwhere(label == 2)

        elif self.sampling_mode == "cyst":
            target_voxels = np.argwhere(label == 3)

        # -----------------------------------------------------
        # Safety fallback
        # -----------------------------------------------------

        if len(target_voxels) == 0:

            # If this case does not contain the requested
            # structure, fall back to any foreground voxel.
            target_voxels = np.argwhere(label > 0)

        if len(target_voxels) == 0:

            # Extremely unlikely for KiTS23, but handle
            # completely empty masks safely.
            d_start = np.random.randint(0, d - pd + 1)
            h_start = np.random.randint(0, h - ph + 1)
            w_start = np.random.randint(0, w - pw + 1)

        else:

            # Random target voxel
            center = target_voxels[
                np.random.randint(len(target_voxels))
            ]

            center_d, center_h, center_w = center

            # Center patch around target voxel
            d_start = center_d - pd // 2
            h_start = center_h - ph // 2
            w_start = center_w - pw // 2

            # Keep patch inside volume
            d_start = max(0, min(d_start, d - pd))
            h_start = max(0, min(h_start, h - ph))
            w_start = max(0, min(w_start, w - pw))

        # -----------------------------------------------------
        # Extract patch
        # -----------------------------------------------------

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