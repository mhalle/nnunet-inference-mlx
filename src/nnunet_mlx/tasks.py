"""
TotalSegmentator task definitions.

Each task maps to a dataset ID used to locate model weights.
"""

from enum import IntEnum


class Task(IntEnum):
    """TotalSegmentator model tasks.

    The int value is the nnU-Net dataset/task ID used to locate weights
    on disk (e.g. Dataset297_*/). Use with ModelBundle.from_task():

        bundle = ModelBundle.from_task(Task.TOTAL_FAST)
    """

    # --- Full body CT (1.5mm, 5 models) ---
    ORGANS = 291
    VERTEBRAE = 292
    CARDIAC = 293
    MUSCLES = 294
    RIBS = 295

    # --- Full body CT (3mm, single model) ---
    TOTAL_FAST = 297

    # --- Full body CT (1.5mm highres, single model) ---
    TOTAL = 298

    # --- Body region / composition ---
    BODY = 300
    BODY_MR = 598
    TISSUE_TYPES = 481
    TISSUE_TYPES_MR = 925
    TISSUE_4_TYPES = 485
    ABDOMINAL_MUSCLES = 952

    # --- Head and neck ---
    HEAD_GLANDS_CAVITIES = 775
    HEADNECK_BONES_VESSELS = 776
    HEAD_MUSCLES = 777
    HEADNECK_MUSCLES_A = 778
    HEADNECK_MUSCLES_B = 779
    OCULOMOTOR_MUSCLES = 351
    FACE = 303
    FACE_MR = 856
    BRAIN_STRUCTURES = 409
    CRANIOFACIAL_STRUCTURES = 115
    TEETH = 113

    # --- Cardiac ---
    HEARTCHAMBERS_HIGHRES = 301
    CORONARY_ARTERIES = 509
    AORTIC_SINUSES = 920

    # --- Lung ---
    LUNG_VESSELS = 117
    LUNG_NODULES = 913

    # --- Liver ---
    LIVER_VESSELS = 8
    LIVER_SEGMENTS = 570
    LIVER_SEGMENTS_MR = 576
    LIVER_LESIONS = 591
    LIVER_LESIONS_MR = 589

    # --- Spine ---
    VERTEBRAE_MR = 756
    VERTEBRAE_BODY = 305

    # --- Musculoskeletal ---
    APPENDICULAR_BONES = 304
    APPENDICULAR_BONES_MR = 855
    THIGH_SHOULDER_MUSCLES = 857
    HIP_IMPLANT = 260

    # --- Other ---
    CEREBRAL_BLEED = 150
    PLEURAL_PERICARD_EFFUSION = 315
    KIDNEY_CYSTS = 789
    BREASTS = 527
    VENTRICLE_PARTS = 552
    TRUNK_CAVITIES = 343
    BRAIN_ANEURYSM = 615


# Multi-model tasks (require multiple task IDs)
TOTAL_FULL = [Task.ORGANS, Task.VERTEBRAE, Task.CARDIAC, Task.MUSCLES, Task.RIBS]
HEADNECK_MUSCLES = [Task.HEADNECK_MUSCLES_A, Task.HEADNECK_MUSCLES_B]
