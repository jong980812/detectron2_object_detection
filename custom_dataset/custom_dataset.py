import torch, detectron2

print(torch.cuda.device_count())
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from data_function import *
from detectron2.engine import DefaultTrainer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
#basket_dicts=get_sports_dicts("/local_datasets/detectron2/toy_example/basketball_sample_json")
#! json 모여있는 곳에 가서 하나로 합친 객체를 던져준다.

#@ Register train dataset@@@@@@@@@@@@@@@@@@@@@@@@@@@
for d in ["train","val"]:
    DatasetCatalog.register(d, lambda d=d: get_sports_dicts("/local_datasets/detectron2/basketball/annotations/"+d+"_json") )
    MetadataCatalog.get(d).set(thing_classes=["Stadium","Three_Point_Line","Paint_zone","Player","Goal_post","Game_tool"])
basketball_metadata = MetadataCatalog.get("train")
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#? Visualization Test @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# for d in random.sample(basket_dicts, 1):
#     img = cv2.imread(os.path.join("/local_datasets/detectron2/toy_example/basketball_sample_json",d["file_name"]))
#     visualizer = Visualizer(img[:, :, ::-1], metadata=basketball_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imwrite("./sample.jpg",(out.get_image()[:, :, ::-1]))
#? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


os.makedirs("./output", exist_ok=True)
def main():
    cfg = get_cfg()
    cfg.merge_from_file("/data/jong980812/nia/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ('train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("/data/jong980812/nia/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 64  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
    cfg.SOLVER.MAX_ITER = 50000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR="./output"
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    # Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg,"val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    return predictor


if __name__=="__main__":
    args = default_argument_parser().parse_args()
    print("start")
    print(f"Device Count{torch.cuda.device_count()}")
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
    )
   