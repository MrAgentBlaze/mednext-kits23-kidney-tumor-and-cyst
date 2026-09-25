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
            "random",
            "tumor",
            "cyst",
            "mixed",
            "balanced",
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

        self.cyst_cases = None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):

        # Determine what type of patch to sample.
        sampling_mode = self._choose_sampling_mode()

        # For cyst sampling, select only from cases
        # that actually contain cyst voxels.
        if sampling_mode == "cyst":

            if self.cyst_cases is None:

                self.cyst_cases = []

                for case_dir in self.cases:

                    label_path = (
                        case_dir / "segmentation.nii.gz"
                    )

                    label_data = nib.load(
                        str(label_path)
                    ).get_fdata()

                    if np.any(label_data == 3):
                        self.cyst_cases.append(
                            case_dir
                        )

                if len(self.cyst_cases) == 0:
                    raise RuntimeError(
                        "No cyst-containing cases found."
                    )

            case_dir = self.cyst_cases[
                np.random.randint(len(self.cyst_cases))
            ]

        else:

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

            # If cyst sampling was selected, make sure
            # the selected case actually contains cyst voxels.
            if sampling_mode == "cyst":

                if not np.any(label == 3):

                    case_dir = self.cyst_cases[
                        np.random.randint(
                            len(self.cyst_cases)
                        )
                    ]

                    continue

            break

        # Basic CT normalization.
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000

        # Extract a patch.
        image_patch, label_patch = self._extract_patch(
            image,
            label,
            sampling_mode=sampling_mode,
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

    def _extract_patch(self, image, label, sampling_mode="foreground"):

        patch_depth, patch_height, patch_width = self.patch_size

        depth, height, width = image.shape

        if height < patch_height or width < patch_width:
            raise ValueError(
                f"Image too small for patch size {self.patch_size}: "
                f"{image.shape}"
            )

        # Pad shallow volumes in depth.
        if depth < patch_depth:
            pad_before = (patch_depth - depth) // 2
            pad_after = patch_depth - depth - pad_before

            image = np.pad(
                image,
                (
                    (pad_before, pad_after),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )

            label = np.pad(
                label,
                (
                    (pad_before, pad_after),
                    (0, 0),
                    (0, 0),
                ),
                mode="constant",
                constant_values=0,
            )

            depth = image.shape[0]

        # ---------------------------------------------------------
        # TRUE RANDOM PATCH
        # ---------------------------------------------------------
        if sampling_mode == "random":

            start_d = np.random.randint(
                0,
                depth - patch_depth + 1,
            )

            start_h = np.random.randint(
                0,
                height - patch_height + 1,
            )

            start_w = np.random.randint(
                0,
                width - patch_width + 1,
            )

        else:

            # Determine which class should guide patch selection.
            if sampling_mode == "tumor":
                target_voxels = np.argwhere(label == 2)

            elif sampling_mode == "cyst":
                target_voxels = np.argwhere(label == 3)

            elif sampling_mode == "foreground":
                target_voxels = np.argwhere(label > 0)

            else:
                raise ValueError(
                    f"Unknown sampling mode: {sampling_mode}"
                )

            # If the requested target does not exist,
            # fall back to any foreground voxel.
            if len(target_voxels) == 0:
                target_voxels = np.argwhere(label > 0)

            # If there is still no foreground,
            # use a completely random patch.
            if len(target_voxels) == 0:

                start_d = np.random.randint(
                    0,
                    depth - patch_depth + 1,
                )

                start_h = np.random.randint(
                    0,
                    height - patch_height + 1,
                )

                start_w = np.random.randint(
                    0,
                    width - patch_width + 1,
                )

            else:

                # Select a random target voxel.
                center_d, center_h, center_w = target_voxels[
                    np.random.randint(len(target_voxels))
                ]

                # Center the patch around that voxel.
                start_d = center_d - patch_depth // 2
                start_h = center_h - patch_height // 2
                start_w = center_w - patch_width // 2

                # Clamp to valid patch boundaries.
                start_d = max(
                    0,
                    min(start_d, depth - patch_depth),
                )

                start_h = max(
                    0,
                    min(start_h, height - patch_height),
                )

                start_w = max(
                    0,
                    min(start_w, width - patch_width),
                )

        end_d = start_d + patch_depth
        end_h = start_h + patch_height
        end_w = start_w + patch_width

        image_patch = image[
            start_d:end_d,
            start_h:end_h,
            start_w:end_w,
        ]

        label_patch = label[
            start_d:end_d,
            start_h:end_h,
            start_w:end_w,
        ]

        return image_patch, label_patch

    def _choose_sampling_mode(self):

        if self.sampling_mode != "balanced":
            return self.sampling_mode

        return np.random.choice(
            ["random", "foreground", "tumor", "cyst"],
            p=[0.10, 0.20, 0.35, 0.35],
        )