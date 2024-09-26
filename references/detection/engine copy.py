import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import detection_utils
from coco_eval import CocoEvaluator
from detection_coco_utils import get_coco_api_from_dataset


# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )

#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         loss_value = losses_reduced.item()

#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             losses.backward()
#             optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return metric_logger

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = detection_utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    # 메트릭 초기화
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]


        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
        print(loss_dict)  # 모델의 출력 내용을 확인합니다.
        # 역전파
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        # 메트릭 누적
        running_loss += losses.item() * len(images)
        
        # 정확도 계산 (필요에 따라 수정)
        model.eval()
        outputs = model(images)
        print(outputs)
        preds = outputs['scores'].argmax(dim=1)  # 각 이미지에 대해 예측된 클래스

        # 실제 라벨과 비교하여 올바른 예측 수 카운트
        labels = torch.cat([target['labels'] for target in targets])  # 각 배치에 대한 실제 라벨

        # preds와 labels의 크기가 같아야 함
        if preds.shape[0] == labels.shape[0]:
            running_corrects += (preds == labels).sum().item()  # 올바른 예측 수
        
        # # 정확도 계산
        # preds = loss_dict['scores'].argmax(dim=1)  # 예측된 클래스
        # labels = torch.cat([target['labels'] for target in targets])  # 실제 라벨

        # # preds와 labels의 크기가 같아야 함
        # if preds.shape[0] == labels.shape[0]:
        #     running_corrects += (preds == labels).sum().item()  # 올바른 예측 수
            
        total_samples += len(images)

        metric_logger.update(loss=losses)

    # 평균 손실 및 정확도 계산
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples  # 정확도 계산 방법에 따라 수정
    
    return epoch_loss,epoch_acc
    

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)     
#     with torch.no_grad():
#         for images, targets in metric_logger.log_every(data_loader, 100, header):
#             images = list(img.to(device) for img in images)         
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             model_time = time.time()
#             outputs = model(images)         
#             outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#             model_time = time.time() - model_time         
#             res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#             evaluator_time = time.time()
#             coco_evaluator.update(res)
#             evaluator_time = time.time() - evaluator_time
#             metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator
def evaluate(model, data_loader_test, device):
    model.eval()  # 평가 모드로 설정
    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    # 손실값을 추적하기 위한 변수
    running_loss = 0.0
    total_samples = 0

    # 평가 모드에서 gradient 계산을 하지 않음
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            # 1. 일시적으로 모델을 훈련 모드로 변경하여 손실 계산
            model.train()  # 훈련 모드로 전환
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            model.eval()  # 다시 평가 모드로 전환

            # 배치 크기 계산
            batch_size = len(images)
            running_loss += losses.item() * batch_size
            total_samples += batch_size

            # 2. 모델의 예측값을 사용하여 평가 진행 (mAP, IoU 계산)
            outputs = model(images)
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

    # 평균 손실 계산
    epoch_loss = running_loss / total_samples

    # IoU와 mAP 등 평가 결과 계산
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return epoch_loss, coco_evaluator


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.box_regression_loss = torch.nn.SmoothL1Loss()
        self.classification_loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        total_loss = 0.0
        classification_loss = self.classification_loss(outputs['scores'], targets['labels'])
        total_loss += classification_loss
        box_regression_loss = self.box_regression_loss(outputs['boxes'], targets['boxes'])
        total_loss += box_regression_loss
        return total_loss

def evaluate0(model, data_loader, device):
    model.eval()
    metric_logger = detection_utils.MetricLogger(delimiter="  ")

    # 메트릭 초기화
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    custom_loss = CustomLoss()
    
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, "Test:"):
            images = list(image.to(device) for image in images)
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            outputs = model(images)  # 추론 모드
            # 손실 및 예측 계산
            losses = custom_loss(outputs, targets)
            
            preds = torch.argmax(outputs['scores'], dim=1)  # 클래스 예측
            labels = torch.cat([target['labels'] for target in targets])  # 정답 라벨
            # 정확도 계산 (필요에 따라 수정)
            running_corrects += (preds == labels).sum().item()  # 직접 비교하여 맞춘 개수 계산

            running_loss += losses.item() * len(images)
            total_samples += len(images)

            metric_logger.update(loss=losses)

    # 평균 손실 및 정확도 계산
    val_loss = running_loss / total_samples
    val_acc = running_corrects / total_samples  # 정확도 계산 방법에 따라 수정

    return val_loss, val_acc
