# ========================== IMPORTS ==========================
# PyTorch & TorchVision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.ops import nms
# Utilities
import os, gc, time, shutil, zipfile, json, warnings, pathlib, sys, io
from collections import defaultdict
from contextlib import redirect_stdout
import random
# Image processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageDraw
import albumentations as A
# PyCOCO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# Progress bar
from tqdm import tqdm
# Ignore Warnings
warnings.filterwarnings('ignore')

# ========================== CONFIG ==========================
run_number = 10
batch_size = 6
num_epochs = 20
learning_rate = 0.0002
enable_augmentation = True
enable_mosaic = True
early_stopping_patience = 10
min_improvement = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================== DATA ==========================
train_dir = pathlib.Path("./v4i.coco/train")
valid_dir = pathlib.Path("./v4i.coco/valid")
test_dir = pathlib.Path("./v4i.coco/test")
annotation_filename = "_annotations.coco_cleaned.json"
train_anno = train_dir / annotation_filename
valid_anno = valid_dir / annotation_filename
test_anno = test_dir / annotation_filename
model_save_dir = pathlib.Path("./models")
model_save_dir.mkdir(parents=True, exist_ok=True)

# reformat the annotations and get tensors
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annotation_file, train=True, augment_fn=None, enable_mosaic=True, mosaic_prob=1):
        with redirect_stdout(io.StringIO()):
            super().__init__(root, annotation_file, transforms=None)
        self.train = train
        self.augment_fn = augment_fn
        self.enable_mosaic = enable_mosaic and train
        self.mosaic = MosaicAugmentation(mosaic_prob=mosaic_prob) if self.enable_mosaic else None

    def get_raw_item(self, index):
        """Get raw image and target without any augmentations"""
        img, target = super().__getitem__(index)
        img = np.array(img)

        # convert bounding box coordinates into (x1, y1, x2, y2) format
        segments = []
        cls = []
        for obj in target:
            x1, y1, w, h = obj['bbox']
            segments.append([x1, y1, x1+w, y1+h])
            cls.append(obj['category_id'])

        return {'img': img, 'segments': segments, 'cls': cls, 'img_index': index}

    def __getitem__(self, index):
        """
        Data loader loads an image, convert bbox coordinates, apply mosaic and other augmentations, convert to tensors.
        """
        # apply mosaic first
        if self.enable_mosaic and self.mosaic and random.random() < self.mosaic.mosaic_prob:
            try:
                mosaic_result = self.mosaic(self, index)
                if mosaic_result is not None:
                    labels = mosaic_result
                else:
                    # fallback to regular loading if mosaic is turned off
                    labels = self.get_raw_item(index)
            except Exception as e:
                print(f"Mosaic failed, using regular image: {e}")
                labels = self.get_raw_item(index)
        else:
            labels = self.get_raw_item(index)
        
        # other augmentations
        if self.augment_fn:
            labels = self.augment_fn(labels)

        # convert to tensors
        return self.convert_to_tensors(labels, index)
    
    def convert_to_tensors(self, labels, index):
        # convert image to tensor
        img_tensor = torch.from_numpy(labels['img']).permute(2,0,1).float() / 255.0
        
        # convert bbox, labels, iscrowd to tensors
        if len(labels['segments']) > 0:
            boxes = torch.as_tensor(labels['segments'], dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels['cls'], dtype=torch.int64)
            areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
            iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            # if no objects, create empty tensors
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
        
        # formatted annotation
        formatted_target = {
            'boxes': boxes,
            'labels': labels_tensor,
            'areas': areas,
            'iscrowd': iscrowd,
            'image_id': torch.tensor([index])
        }
                
        return img_tensor, formatted_target

# data collator
def collate_fn(batch):
       images, targets = zip(*batch)
       return torch.stack(images), list(targets)

# ========================== DATA AUGMENTATION ==========================
class MosaicAugmentation:
    def __init__(self, mosaic_prob=1):
        self.mosaic_prob = mosaic_prob

    def __call__(self, dataset, index):
        
        # get 4 images for mosaic
        indices = [index]
        while len(indices) < 4:
            rand_idx = random.randint(0, len(dataset) - 1)
            if rand_idx not in indices:
                indices.append(rand_idx)

        # load images, bboxes, and labels using raw data access
        images = []
        all_segments = []
        all_classes = []

        for idx in indices:
            # Use get_raw_item to avoid recursion
            raw_data = dataset.get_raw_item(idx)
            
            images.append(raw_data['img'])
            all_segments.append(raw_data['segments'])
            all_classes.append(raw_data['cls'])

        # use dimensions from first image (assuming all images same size)
        img_h, img_w = images[0].shape[:2]

        mosaic_img, mosaic_segments, mosaic_classes = self.create_mosaic_4(
            images, all_segments, all_classes, img_h, img_w, min_visibility=0.7
        )

        return {
            'img': mosaic_img,
            'segments': mosaic_segments,
            'cls': mosaic_classes,
            'img_index': index
        }
    
    def create_mosaic_4(self, images, all_segments, all_classes, img_h, img_w, min_visibility=0.7):
        # stitch 4 images into a 2x2 grid
        stitched_h = img_h * 2
        stitched_w = img_w * 2
        stitched_img = np.full((stitched_h, stitched_w, 3), 0, dtype=np.uint8)
        stitched_segments = []
        stitched_classes = []
        
        for i, (img, segments, classes) in enumerate(zip(images, all_segments, all_classes)):
            h, w = img.shape[:2]
            
            # calculate position in stitched image
            if i == 0:  # top left
                y_offset, x_offset = 0, 0
            elif i == 1:  # top right
                y_offset, x_offset = 0, img_w
            elif i == 2:  # bottom left
                y_offset, x_offset = img_h, 0
            elif i == 3:  # bottom right
                y_offset, x_offset = img_h, img_w
            
            # place image in stitched grid
            stitched_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
            # transform bounding boxes to stitched coordinates
            for segment, cls in zip(segments, classes):
                if len(segment) == 4:
                    x1, y1, x2, y2 = segment
                    stitched_segments.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset])
                    stitched_classes.append(cls)
        
        # choose random center point and crop to original size
        yc = int(random.uniform(0.5 * img_h, 1.5 * img_h))
        xc = int(random.uniform(0.5* img_w, 1.5 * img_w))

        # calculate crop boundaries (original image size)
        crop_x1 = xc - img_w // 2
        crop_y1 = yc - img_h // 2
        crop_x2 = crop_x1 + img_w
        crop_y2 = crop_y1 + img_h
        
        # ensure crop boundaries are valid
        crop_x1 = max(0, min(crop_x1, stitched_w - img_w))
        crop_y1 = max(0, min(crop_y1, stitched_h - img_h))
        crop_x2 = crop_x1 + img_w
        crop_y2 = crop_y1 + img_h
        
        # crop the stitched image
        final_img = stitched_img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # transform and filter bounding boxes
        final_segments = []
        final_classes = []
        
        for segment, cls in zip(stitched_segments, stitched_classes):
            x1, y1, x2, y2 = segment
            original_area = (x2 - x1) * (y2 - y1)
            
            # transform to cropped coordinates
            x1_new = x1 - crop_x1
            y1_new = y1 - crop_y1
            x2_new = x2 - crop_x1
            y2_new = y2 - crop_y1
            
            # clip to final image boundaries
            x1_final = max(0, min(x1_new, img_w))
            y1_final = max(0, min(y1_new, img_h))
            x2_final = max(0, min(x2_new, img_w))
            y2_final = max(0, min(y2_new, img_h))
            
            # check visibility and size filters
            if x2_final > x1_final and y2_final > y1_final:
                final_area = (x2_final - x1_final) * (y2_final - y1_final)
                visibility = final_area / original_area if original_area > 0 else 0
                
                if (x2_final - x1_final) > 10 and (y2_final - y1_final) > 10 and visibility >= min_visibility:
                    final_segments.append([x1_final, y1_final, x2_final, y2_final])
                    final_classes.append(cls)
        
        return final_img, final_segments, final_classes


class DataAugmentation:
    def __init__(self, enable_augmentation=True, enable_mosaic=True, mosaic_prob=1):
        self.enable_augmentation = enable_augmentation
        self.enable_mosaic = enable_mosaic
        self.mosaic = MosaicAugmentation(mosaic_prob=mosaic_prob) if enable_mosaic else None
        
        # augmentation pipeline (Alblumentations) - order: geometric, color, then flip
        if enable_augmentation:
            self.transform = A.Compose([
                A.Affine(scale=(0.5, 1.5), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=1),
                A.ColorJitter(brightness=0.4, saturation=0.7, hue=0.015, p=1),
                A.HorizontalFlip(p=0.5)
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels'], 
                min_visibility=0.7 # keep bbox only if more than 70% of the object is available
            ))

    def __call__(self, labels):
        # if no augmentation is to be applied, just return the original dict
        if not self.enable_augmentation or len(labels['segments']) == 0:
            return labels
        
        img = labels['img']
        segments = labels['segments']
        cls = labels['cls']

        # apply augmentation
        try:
            augmented = self.transform(image=img, bboxes=segments, class_labels=cls)
            labels['img'] = augmented['image']
            labels['segments'] = augmented['bboxes']
            labels['cls'] = augmented['class_labels']
        except Exception as e:
            print(f"Augmentation Failed: {e}")
        
        return labels

# ========================== MODEL SETUP ==========================
def get_num_classes(annotation_file):
       with open(annotation_file, 'r') as f:
              data = json.load(f)
       categories = data.get('categories', [])
       max_id = 0
       for category in categories:
              print(f"ID: {category['id']}, Name: {category['name']}")
              max_id = max(max_id, category['id'])  # finds how many classes there are in the annotation
       num_classes = max_id +1
       print(f"Number of classes (including background): {num_classes}")
       return num_classes

def get_faster_rcnn(num_classes):
       model = fasterrcnn_resnet50_fpn(pretrained=True)
       in_features = model.roi_heads.box_predictor.cls_score.in_features
       model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
       total_params = sum(p.numel() for p in model.parameters())
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       print(f"Model accessed from TorchVision")
       print(f"Total parameters: {total_params:,} / Trainable parameters: {trainable_params:,}")
       return model


# ========================== TRAINING UTILITIES ==========================
class AverageMeter:
       def __init__(self):
              self.reset()
       def reset(self):
              self.val = 0
              self.avg = 0
              self.sum = 0
              self.count = 0
       def update(self, val, n):
              self.val = val
              self.sum += val * n
              self.count += n
              self.avg = self.sum / self.count

# inner training loop              
def train_one_epoch(model, optimizer, data_loader, device, epoch_no, total_epochs):
       model.train()

       # initializie loss
       total_loss_meter = AverageMeter()
       classifier_loss_meter = AverageMeter()
       box_reg_loss_meter = AverageMeter()
       objectness_loss_meter = AverageMeter()
       rpn_box_reg_loss_meter = AverageMeter()  

       with tqdm(data_loader, unit='batch') as tepoch:
              tepoch.set_description(f"Train:Epoch {epoch_no}/{total_epochs}")    

              for images, targets in tepoch:
                     images = [img.to(device) for img in images]
                     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                     # get loss values
                     loss_dict = model(images, targets)
                     classifier_loss = loss_dict.get("loss_classifier", torch.tensor(0.0)).cpu().detach().numpy()
                     box_reg_loss = loss_dict.get("loss_box_reg", torch.tensor(0.0)).cpu().detach().numpy()
                     objectness_loss = loss_dict.get("loss_objectness", torch.tensor(0.0)).cpu().detach().numpy()
                     rpn_box_reg_loss = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).cpu().detach().numpy()
                     total_loss = sum(loss for loss in loss_dict.values())

                     # backward pass
                     optimizer.zero_grad()
                     total_loss.backward()
                     optimizer.step()

                     # update meters
                     batch_size = len(images)
                     total_loss_meter.update(total_loss.item(), batch_size)
                     classifier_loss_meter.update(classifier_loss, batch_size)
                     box_reg_loss_meter.update(box_reg_loss, batch_size)
                     objectness_loss_meter.update(objectness_loss, batch_size)
                     rpn_box_reg_loss_meter.update(rpn_box_reg_loss, batch_size)

                     # update progress bar
                     tepoch.set_postfix(
                            total_loss=f"{total_loss_meter.avg:.4f}",
                            cls_loss=f"{classifier_loss_meter.avg:.4f}",
                            box_loss=f"{box_reg_loss_meter.avg:.4f}",
                            obj_loss=f"{objectness_loss_meter.avg:.4f}",
                            rpn_loss=f"{rpn_box_reg_loss_meter.avg:.4f}"
                     )
                     
       return {
              'total_loss': total_loss_meter.avg,
              'classifier_loss': classifier_loss_meter.avg,
              'box_reg_loss': box_reg_loss_meter.avg,
              'objectness_loss': objectness_loss_meter.avg,
              'rpn_box_reg_loss': rpn_box_reg_loss_meter.avg
       }

# validation loop (same function with above), just without gradient updates
@torch.inference_mode()
def evaluate(model, data_loader, device, epoch_no, total_epochs):
       model.train()
       total_loss_meter = AverageMeter()
       classifier_loss_meter = AverageMeter()
       box_reg_loss_meter = AverageMeter()
       objectness_loss_meter = AverageMeter()
       rpn_box_reg_loss_meter = AverageMeter()  
       with tqdm(data_loader, unit='batch') as tepoch:
              tepoch.set_description(f"Val:Epoch {epoch_no}/{total_epochs}")    
              for images, targets in tepoch:
                     images = [img.to(device) for img in images]
                     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                     loss_dict = model(images, targets)
                     classifier_loss = loss_dict.get("loss_classifier", torch.tensor(0.0)).cpu().detach().numpy()
                     box_reg_loss = loss_dict.get("loss_box_reg", torch.tensor(0.0)).cpu().detach().numpy()
                     objectness_loss = loss_dict.get("loss_objectness", torch.tensor(0.0)).cpu().detach().numpy()
                     rpn_box_reg_loss = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).cpu().detach().numpy()
                     total_loss = sum(loss for loss in loss_dict.values())
                     batch_size = len(images)
                     total_loss_meter.update(total_loss.item(), batch_size)
                     classifier_loss_meter.update(classifier_loss, batch_size)
                     box_reg_loss_meter.update(box_reg_loss, batch_size)
                     objectness_loss_meter.update(objectness_loss, batch_size)
                     rpn_box_reg_loss_meter.update(rpn_box_reg_loss, batch_size)
                     tepoch.set_postfix(
                            total_loss=f"{total_loss_meter.avg:.4f}",
                            cls_loss=f"{classifier_loss_meter.avg:.4f}",
                            box_loss=f"{box_reg_loss_meter.avg:.4f}",
                            obj_loss=f"{objectness_loss_meter.avg:.4f}",
                            rpn_loss=f"{rpn_box_reg_loss_meter.avg:.4f}"
                     )    
       return {
              'total_loss': total_loss_meter.avg,
              'classifier_loss': classifier_loss_meter.avg,
              'box_reg_loss': box_reg_loss_meter.avg,
              'objectness_loss': objectness_loss_meter.avg,
              'rpn_box_reg_loss': rpn_box_reg_loss_meter.avg
       }

# outer training loop
def model_training(model, num_epochs, min_improvement, early_stopping_patience, train_loader, val_loader, optimizer, lr_scheduler, device, model_save_dir):
       # initialize tracking variables
       train_losses = []
       val_losses = []
       learning_rates = []
       best_val_loss = float('inf')
       best_model_state = None
       best_epoch = 0
       patience_counter = 0
       print("Start training...")

       # training
       for epoch in range(num_epochs):
              train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch+1, num_epochs)
              train_losses.append(train_metrics['total_loss'])
              val_metrics = evaluate(model, val_loader, device, epoch+1, num_epochs)
              val_losses.append(val_metrics['total_loss'])

              # LR update
              lr_scheduler.step()
              current_lr = lr_scheduler.get_last_lr()[0]
              learning_rates.append(current_lr)

              # epoch summary
              print(f"Epoch {epoch+1}/{num_epochs} Summary:")
              print(f"Train Loss: {train_metrics['total_loss']:.4f}, Val Loss: {val_metrics['total_loss']:.4f}, LR: {current_lr:.6f}")
              print(f"Train - Cls: {train_metrics['classifier_loss']:.4f}, Box: {train_metrics['box_reg_loss']:.4f}, Obj: {train_metrics['objectness_loss']:.4f}, RPN: {train_metrics['rpn_box_reg_loss']:.4f}")
              print(f"Val   - Cls: {val_metrics['classifier_loss']:.4f}, Box: {val_metrics['box_reg_loss']:.4f}, Obj: {val_metrics['objectness_loss']:.4f}, RPN: {val_metrics['rpn_box_reg_loss']:.4f}")

              # check model improvement
              improvement = best_val_loss - val_metrics['total_loss']
              if improvement > min_improvement:
                     best_val_loss = val_metrics['total_loss']
                     best_epoch = epoch + 1
                     best_model_state = model.state_dict().copy()  # store in memory
                     patience_counter = 0
                     print(f'New best model.')
              else:
                     patience_counter += 1
                     print(f'No improvement for {patience_counter} epoch(s)')

              # trigger early stopping       
              if patience_counter >= early_stopping_patience:
                     print(f'\nEarly stopping triggered - no improvement for {early_stopping_patience} epochs')
                     break
              
              print()

       if best_model_state is not None:
              torch.save(best_model_state, model_save_dir / f'{run_number}best_model.pt')
              print(f"\nBest model saved in {model_save_dir}")

       # plot training progress summary
       plt.figure(figsize=(5, 4))
       epochs = range(1, len(train_losses) + 1)
       plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
       plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.title('Training and Validation Loss')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.xticks(epochs)
       plt.tight_layout()
       plt.show()

       return {
              'train_losses': train_losses,
              'val_losses': val_losses,
              'learning_rates': learning_rates,
              'best_epoch': best_epoch,
              'best_val_loss': best_val_loss
       }


# ========================== COCO EVAL ==========================
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

@torch.inference_mode() # conf threshold is default to 0.001 for YOLO val and 0.25 for YOLO predict 
def coco_eval(model, data_loader, annotation_file, device, score_threshold=0.001):   
       # load ground truth labels
       cocoGt = COCO(annotation_file)

       # get image IDs
       imgIds = cocoGt.getImgIds()

       # evaluation
       model.eval()
       inference_times = []
       res_id = 1
       res_all = []

       for i, (image, target) in enumerate(tqdm(data_loader)):
              image_batch = image.to(device)

              start_time = time.time()
              predictions = model(image_batch)
              if device.type == 'cuda':
                     torch.cuda.synchronize()
              end_time = time.time()
              inference_times.append(end_time - start_time)
              pred = predictions[0]

              img_id = imgIds[i] if i < len(imgIds) else i

              # filter out preditions with low score
              keep_idx = pred['scores'] >= score_threshold
              if keep_idx.sum() > 0:
                     pred_boxes = pred['boxes'][keep_idx].cpu().numpy()
                     pred_scores = pred['scores'][keep_idx].cpu().numpy()
                     pred_labels = pred['labels'][keep_idx].cpu().numpy()

                     pred_boxes_xywh = []
                     for box in pred_boxes:
                            x1, y1, x2, y2 = box
                            w, h = x2-x1, y2-y1
                            pred_boxes_xywh.append([x1, y1, w, h])

                     for j in range(len(pred_boxes_xywh)):
                            res_temp = {
                                   "id": res_id, 
                                   "image_id": int(img_id),
                                   "bbox": [float(x) for x in pred_boxes_xywh[j]],
                                   "segmentation": [],
                                   "iscrowd": 0,
                                   "category_id": int(pred_labels[j]),
                                   "area": float(pred_boxes_xywh[j][2] * pred_boxes_xywh[j][3]),
                                   "score": float(pred_scores[j])
                            }
                            res_all.append(res_temp)
                            res_id += 1

       # write results into a json file
       save_json_path = 'temp_predictions.json'
       save_json(res_all, save_json_path)

       # run COCO eval on saved json file
       stats = None
       try:
              cocoDt = cocoGt.loadRes(save_json_path)
              cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
              cocoEval.evaluate()
              cocoEval.accumulate()
              cocoEval.summarize()
              stats = cocoEval.stats

       except Exception as e:
              print(f"COCO eval failed: {e}")

       # get rid of saved json
       if os.path.exists(save_json_path):
              os.remove(save_json_path)

       # display performance
       avg_time = np.mean(inference_times)
       print(f"\nPerformance Metrics:")
       print(f"Average inference time: {avg_time*1000:.2f} ms")
       print(f"Throughput: {1/avg_time:.1f} images per second")
       print(f"mAP0.5: {stats[1]:.4f}")
       print(f"mAP0.5-0.95: {stats[0]:.4f}")

       # clean up memory
       torch.cuda.empty_cache()
       gc.collect()

       return stats, res_all


# ========================== VISUALIZATION ==========================
def visualize_augmented_data(data_loader, num_images=5, figsize=(15, 5)):
    batch_images, batch_targets = next(iter(data_loader))
    fig, axs = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 1:
        axs = [axs]
    
    for i in range(min(num_images, len(batch_images))):
        img_tensor = batch_images[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax = axs[i]
        
        ax.imshow(img_np)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
        
        target = batch_targets[i]
        
        for box in target['boxes']:
            x1, y1, x2, y2 = box.tolist() # getting from dataset so format is x1, y1, x2, y2
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            ) 
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()


# ========================== INFERENCE AND SHOW ==========================
@torch.inference_mode()
def predict(model, data_loader, device, num_images=5, score_threshold=0.25, figsize=(15, 5), nms_threshold=0.15):
       model.eval()
       fig, axs = plt.subplots(1, num_images, figsize=figsize)
       if num_images == 1:
              axs = [axs]

       images_processed = 0

       for batch_images, batch_targets in data_loader:
              for i in range(len(batch_images)):

                     # get images to GPU and run forward pass
                     img_tensor = batch_images[i]
                     img_batch = img_tensor.unsqueeze(0).to(device)
                     predictions = model(img_batch)
                     pred = predictions[0]

                     # convert to numpy for plotting
                     img_np = img_tensor.permute(1, 2, 0).numpy()
                     ax = axs[images_processed]

                     ax.imshow(img_np)
                     ax.set_title(f"Prediction {images_processed +1}")
                     ax.axis("off")

                     # filter out predictions lower than threshold conf
                     keep_idx = pred['scores'] >= score_threshold

                     if keep_idx.sum() > 0:
                            pred_boxes = pred['boxes'][keep_idx]
                            pred_scores = pred['scores'][keep_idx]
                            pred_labels = pred['labels'][keep_idx]

                            # apply nms
                            nms_idx = nms(pred_boxes, pred_scores, nms_threshold)

                            nms_boxes = pred_boxes[nms_idx].cpu().numpy()
                            nms_scores = pred_scores[nms_idx].cpu().numpy()
                            nms_labels = pred_labels[nms_idx].cpu().numpy()

                            for j, (box, score, label) in enumerate(zip(nms_boxes, nms_scores, nms_labels)):
                                   x1, y1, x2, y2 = box
                                   w, h = x2 - x1, y2 - y1
                                   rect = patches.Rectangle(
                                          (x1, y1), w, h,
                                          linewidth=2, edgecolor='green', facecolor='none'
                                   )
                                   ax.add_patch(rect)

                                   # add labels
                                   label_text = f"Class {int(label)}: {score:.2f}"
                                   ax.text(x1, y1-5, label_text, color='green', fontsize=8,
                                           bbox=dict(boxstyle='round, pad=0.2', facecolor='white', alpha=0.7))
                                   
                     images_processed += 1
              
              if images_processed >= num_images:
                     break

       plt.tight_layout
       plt.show()


# ========================== MAIN FUNCTION ==========================
def main():
    augment_fn = DataAugmentation()
    train_dataset = CustomCocoDetection(root=train_dir, annotation_file=train_anno, train=True, augment_fn=augment_fn, enable_mosaic=enable_mosaic)
    valid_dataset = CustomCocoDetection(root=valid_dir, annotation_file=valid_anno, train=True, augment_fn=None, enable_mosaic=False)
    test_dataset = CustomCocoDetection(root=test_dir, annotation_file=test_anno, train=True, augment_fn=None, enable_mosaic=False)

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    #visualize_augmented_data(train_loader)

    num_of_classes = get_num_classes(train_anno)
    model = get_faster_rcnn(num_of_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

    training_results = model_training(
        model=model, 
        num_epochs=num_epochs,
        min_improvement=min_improvement,
        early_stopping_patience=early_stopping_patience,
        train_loader=train_loader,
        val_loader=valid_loader, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        device=device,
        model_save_dir=model_save_dir
    )

    test_stats, test_results = coco_eval(
        model=model, 
        data_loader=test_loader,
        annotation_file=test_anno,
        device=device, 
        score_threshold=0.001
    )

    #predict(model, test_loader, device)
    
    return training_results, test_stats, test_results


if __name__ == "__main__":
    main()


