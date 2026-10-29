import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get("OAVAT_DATA_ROOT") or os.path.normpath(
    os.path.join(ROOT_PATH, "..", "..", "..", "data", "oa-vat")
)

DINOV3_WEIGHTS_DIR = os.path.join(DATA_ROOT, "dinov3")
ORTRACK_WEIGHTS_DIR = os.path.join(DATA_ROOT, "ortrack")
YOLO_WEIGHTS_DIR = os.path.join(DATA_ROOT, "yolo")

DINOV3_PATH = os.path.join(ROOT_PATH, "dinov3-main")
ORTRACK_PATH = os.path.join(ROOT_PATH, "ORTrack")

DEFAULT_DINOV3_VITB16 = os.path.join(
    DINOV3_WEIGHTS_DIR, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
DEFAULT_DINOV3_VITS16 = os.path.join(
    DINOV3_WEIGHTS_DIR, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_DINOV3_CHECKPOINT = DEFAULT_DINOV3_VITB16

DEFAULT_ORTRACK_CHECKPOINT = os.path.join(
    ORTRACK_WEIGHTS_DIR, "ORTrack_ep0300.pth.tar"
)

DEFAULT_YOLO_MODEL = os.path.join(YOLO_WEIGHTS_DIR, "yoloe-11l-seg.pt")

MOBILECLIP_PATH = os.path.join(DATA_ROOT, "mobileclip_blt.ts")
