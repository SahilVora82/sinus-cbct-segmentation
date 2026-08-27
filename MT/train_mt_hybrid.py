from pathlib import Path
import csv
import random
import time

import numpy as np
import nrrd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from unet import UNet

BASE_DIR = Path(__file__).resolve().parent
TRAIN_SCAN_DIR = BASE_DIR / 'data' / 'scan'
TRAIN_MASK_DIR = BASE_DIR / 'data' / 'segment'
VAL_SCAN_DIR = BASE_DIR / 'data' / 'scan_valid'
VAL_MASK_DIR = BASE_DIR / 'data' / 'segment_valid'
HEALTHY_CHECKPOINT = BASE_DIR / 'pretrained' / 'healthy_model_50.pth'
OUTPUT_DIR = BASE_DIR / 'results_hybrid'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHANNELS=[32,64,128,256]
NUM_CLASSES=3
LABELS={0:'background',1:'mucosal_thickening',2:'sinus_air'}
TRAIN_MEAN=-46.1730322190273
TRAIN_STD=293.1394271328278
BATCH_SIZE=1
MAX_EPOCHS=50
WARMUP_EPOCHS=2
WARMUP_LR=1e-4
FINETUNE_LR=1e-4
CLASS_WEIGHTS=[1.0,8.0,2.0]
FLIP_PROBABILITY=0.5
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=5
EARLY_STOPPING_PATIENCE=15
MIN_MT_IMPROVEMENT=1e-4
SEED=42
BEST_MT_PATH=OUTPUT_DIR/'best_mt_model.pt'
BEST_FOREGROUND_PATH=OUTPUT_DIR/'best_foreground_model.pt'
LATEST_PATH=OUTPUT_DIR/'latest.pt'
INTERRUPTED_PATH=OUTPUT_DIR/'interrupted.pt'
HISTORY_PATH=OUTPUT_DIR/'training_history.csv'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def get_device():
    if torch.cuda.is_available():
        device=torch.device('cuda'); print('Using CUDA:',torch.cuda.get_device_name(0)); return device
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
        print('Using Apple MPS'); return torch.device('mps')
    print('Using CPU'); return torch.device('cpu')
DEVICE=get_device()

class MTDataset(Dataset):
    def __init__(self, scan_dir: Path, mask_dir: Path, augment: bool):
        self.scan_dir=Path(scan_dir); self.mask_dir=Path(mask_dir); self.augment=augment
        self.scan_files=sorted(self.scan_dir.glob('*.nrrd'))
        if not self.scan_files: raise RuntimeError(f'No scans found in {self.scan_dir}')
        for scan_path in self.scan_files:
            mask_path=self.mask_dir/f'{scan_path.stem}_GT.nrrd'
            if not mask_path.exists(): raise FileNotFoundError(f'Missing mask for {scan_path.name}: {mask_path}')
    def __len__(self): return len(self.scan_files)
    def __getitem__(self,index):
        scan_path=self.scan_files[index]; mask_path=self.mask_dir/f'{scan_path.stem}_GT.nrrd'
        scan,_=nrrd.read(str(scan_path)); mask,_=nrrd.read(str(mask_path))
        if scan.shape!=mask.shape: raise ValueError(f'Shape mismatch for {scan_path.name}: scan={scan.shape}, mask={mask.shape}')
        scan=scan.astype(np.float32); mask=mask.astype(np.int64)
        scan=(scan-TRAIN_MEAN)/TRAIN_STD
        scan=torch.from_numpy(scan).float().unsqueeze(0); mask=torch.from_numpy(mask).long()
        if self.augment and random.random()<FLIP_PROBABILITY:
            scan=torch.flip(scan,dims=[1]); mask=torch.flip(mask,dims=[0])
        return scan,mask

def safe_torch_load(path: Path):
    try: return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(path,map_location='cpu')

def extract_state_dict(checkpoint):
    if not isinstance(checkpoint,dict): raise RuntimeError('Healthy checkpoint is not a dictionary.')
    if 'model_state_dict' in checkpoint: state=checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint: state=checkpoint['state_dict']
    elif 'model' in checkpoint: state=checkpoint['model']
    elif checkpoint and all(isinstance(v,torch.Tensor) for v in checkpoint.values()): state=checkpoint
    else: raise RuntimeError('Could not find model weights in healthy checkpoint.')
    cleaned={}
    for key,value in state.items():
        if key.startswith('module.'): key=key[len('module.'):]
        cleaned[key]=value
    return cleaned

def build_hybrid_model():
    if not HEALTHY_CHECKPOINT.exists(): raise FileNotFoundError(f'Healthy checkpoint not found:\n{HEALTHY_CHECKPOINT}')
    model=UNet(channels=CHANNELS)
    if not hasattr(model,'conv_out'): raise RuntimeError('Expected model.conv_out to exist.')
    if model.conv_out.out_channels!=NUM_CLASSES: raise RuntimeError(f'Current MT U-Net must output {NUM_CLASSES} classes, but conv_out has {model.conv_out.out_channels}.')
    checkpoint=safe_torch_load(HEALTHY_CHECKPOINT); healthy_state=extract_state_dict(checkpoint); mt_state=model.state_dict()
    transferred=[]; skipped=[]
    for key,healthy_value in healthy_state.items():
        if key.startswith('conv_out.'): continue
        if key not in mt_state: skipped.append((key,'missing in MT model')); continue
        if mt_state[key].shape!=healthy_value.shape:
            skipped.append((key,f'shape {tuple(healthy_value.shape)} != {tuple(mt_state[key].shape)}')); continue
        mt_state[key]=healthy_value; transferred.append(key)
    expected_backbone_keys=[k for k in mt_state if not k.startswith('conv_out.')]
    coverage=len(transferred)/max(1,len(expected_backbone_keys))
    if coverage<0.95:
        print('\nSkipped backbone keys:')
        for item in skipped: print(' ',item)
        raise RuntimeError(f'Only {coverage*100:.1f}% of non-output parameters/buffers transferred. Architecture mismatch is too large.')
    hwk='conv_out.weight'; hbk='conv_out.bias'
    if hwk not in healthy_state: raise RuntimeError('Healthy checkpoint has no conv_out.weight.')
    healthy_head_weight=healthy_state[hwk]; mt_head_weight=mt_state[hwk].clone()
    if healthy_head_weight.shape[0]!=1: raise RuntimeError(f'Expected healthy conv_out to have exactly 1 output channel, but got shape {tuple(healthy_head_weight.shape)}.')
    if mt_head_weight.shape[0]!=3: raise RuntimeError(f'Expected MT conv_out to have exactly 3 output channels, but got shape {tuple(mt_head_weight.shape)}.')
    if healthy_head_weight.shape[1:]!=mt_head_weight.shape[1:]: raise RuntimeError('Healthy and MT output heads have incompatible input/kernel dimensions.')
    mt_head_weight.zero_(); mt_head_weight[0].copy_(-0.5*healthy_head_weight[0]); mt_head_weight[1].zero_(); mt_head_weight[2].copy_(0.5*healthy_head_weight[0]); mt_state[hwk]=mt_head_weight
    if hbk in healthy_state and hbk in mt_state:
        healthy_head_bias=healthy_state[hbk]; mt_head_bias=mt_state[hbk].clone()
        if healthy_head_bias.numel()!=1: raise RuntimeError('Expected healthy conv_out.bias to contain 1 value.')
        if mt_head_bias.numel()!=3: raise RuntimeError('Expected MT conv_out.bias to contain 3 values.')
        mt_head_bias.zero_(); mt_head_bias[0]=-0.5*healthy_head_bias[0]; mt_head_bias[1]=0.0; mt_head_bias[2]=0.5*healthy_head_bias[0]; mt_state[hbk]=mt_head_bias
    model.load_state_dict(mt_state,strict=True)
    print('\n'+'='*72); print('HYBRID HEALTHY -> MT INITIALIZATION'); print('='*72)
    print('Healthy checkpoint:',HEALTHY_CHECKPOINT)
    print(f'Backbone parameters/buffers transferred: {len(transferred)}/{len(expected_backbone_keys)}')
    print(f'Backbone transfer coverage: {coverage*100:.1f}%')
    print('Healthy binary output head mapped into:')
    print('  class 0 background = -0.5 × healthy air head'); print('  class 1 MT         = zero-initialized'); print('  class 2 air        = +0.5 × healthy air head'); print('='*72)
    return model

class ForegroundDiceLoss(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,smooth=1e-5): super().__init__(); self.num_classes=num_classes; self.smooth=smooth
    def forward(self,logits,target):
        probabilities=F.softmax(logits,dim=1)
        target_one_hot=F.one_hot(target.long(),num_classes=self.num_classes).permute(0,4,1,2,3).float()
        probabilities=probabilities[:,1:]; target_one_hot=target_one_hot[:,1:]
        dimensions=(0,2,3,4)
        intersection=torch.sum(probabilities*target_one_hot,dim=dimensions)
        predicted_volume=torch.sum(probabilities,dim=dimensions); target_volume=torch.sum(target_one_hot,dim=dimensions)
        dice=(2.0*intersection+self.smooth)/(predicted_volume+target_volume+self.smooth)
        return 1.0-dice.mean()

def hard_class_metrics(predicted_labels,target,class_id):
    predicted=predicted_labels==class_id; truth=target==class_id
    tp=(predicted & truth).sum().item(); fp=(predicted & (~truth)).sum().item(); fn=((~predicted) & truth).sum().item()
    dd=2*tp+fp+fn; iden=tp+fp+fn
    dice=(2*tp)/dd if dd>0 else 1.0; iou=tp/iden if iden>0 else 1.0
    precision=tp/(tp+fp) if (tp+fp)>0 else 0.0; recall=tp/(tp+fn) if (tp+fn)>0 else 1.0
    return {'dice':float(dice),'iou':float(iou),'precision':float(precision),'recall':float(recall)}

def average_metric(metric_list,name): return float(np.mean([item[name] for item in metric_list]))
def freeze_backbone_train_head_only(model):
    for p in model.parameters(): p.requires_grad=False
    for p in model.conv_out.parameters(): p.requires_grad=True
def unfreeze_entire_model(model):
    for p in model.parameters(): p.requires_grad=True

def calculate_combined_loss(logits,target,ce_loss,dice_loss):
    ce=ce_loss(logits,target); dice=dice_loss(logits,target); total=0.5*ce+0.5*dice; return total,ce,dice

def train_one_epoch(model,dataloader,optimizer,ce_loss,dice_loss,head_only,epoch):
    if head_only: model.eval(); model.conv_out.train()
    else: model.train()
    total_loss=total_ce=total_dice=0.0
    for batch_index,(scan,target) in enumerate(dataloader,start=1):
        scan=scan.to(DEVICE,dtype=torch.float32); target=target.to(DEVICE,dtype=torch.long)
        optimizer.zero_grad(set_to_none=True); logits=model(scan); loss,ce,dice=calculate_combined_loss(logits,target,ce_loss,dice_loss)
        loss.backward(); optimizer.step(); total_loss+=loss.item(); total_ce+=ce.item(); total_dice+=dice.item()
        print(f'\rEpoch {epoch:02d}/{MAX_EPOCHS} | batch {batch_index:02d}/{len(dataloader)} | loss {loss.item():.4f}',end='',flush=True)
    print(); n=len(dataloader); return {'loss':total_loss/n,'ce':total_ce/n,'dice_loss':total_dice/n}

def validate_one_epoch(model,dataloader,ce_loss,dice_loss):
    model.eval(); total_loss=total_ce=total_dice=0.0; mt_metrics=[]; air_metrics=[]
    with torch.no_grad():
        for scan,target in dataloader:
            scan=scan.to(DEVICE,dtype=torch.float32); target=target.to(DEVICE,dtype=torch.long)
            logits=model(scan); loss,ce,dice=calculate_combined_loss(logits,target,ce_loss,dice_loss)
            total_loss+=loss.item(); total_ce+=ce.item(); total_dice+=dice.item(); labels=torch.argmax(logits,dim=1)
            for i in range(labels.shape[0]):
                mt_metrics.append(hard_class_metrics(labels[i],target[i],1)); air_metrics.append(hard_class_metrics(labels[i],target[i],2))
    n=len(dataloader); mt_dice=average_metric(mt_metrics,'dice'); air_dice=average_metric(air_metrics,'dice')
    return {'loss':total_loss/n,'ce':total_ce/n,'dice_loss':total_dice/n,'mt_dice':mt_dice,'mt_iou':average_metric(mt_metrics,'iou'),'mt_precision':average_metric(mt_metrics,'precision'),'mt_recall':average_metric(mt_metrics,'recall'),'air_dice':air_dice,'air_iou':average_metric(air_metrics,'iou'),'air_precision':average_metric(air_metrics,'precision'),'air_recall':average_metric(air_metrics,'recall'),'foreground_mean_dice':(mt_dice+air_dice)/2.0}

def make_checkpoint(model,optimizer,scheduler,epoch,phase,train_metrics,val_metrics,best_mt_dice,best_foreground_dice):
    return {'epoch':epoch,'phase':phase,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler is not None else None,'train_metrics':train_metrics,'validation_metrics':val_metrics,'best_mt_dice':best_mt_dice,'best_foreground_dice':best_foreground_dice,'channels':CHANNELS,'labels':LABELS,'class_weights':CLASS_WEIGHTS,'normalization':{'type':'fixed_mt_training_set','mean':TRAIN_MEAN,'std':TRAIN_STD},'augmentation':{'random_flip_axis':0,'probability':FLIP_PROBABILITY},'healthy_checkpoint':str(HEALTHY_CHECKPOINT),'head_initialization':'class0=-0.5*healthy_head, class1=zero, class2=+0.5*healthy_head','warmup_epochs':WARMUP_EPOCHS,'warmup_lr':WARMUP_LR,'finetune_lr':FINETUNE_LR,'test_data_used':False}

def main():
    print('\n'+'='*72); print('HYBRID HEALTHY-PRETRAINED MT TRAINING'); print('='*72)
    print('Device:',DEVICE); print('Architecture:',CHANNELS); print('Labels: 0=background, 1=MT, 2=air')
    print('MT normalization mean/std:',TRAIN_MEAN,TRAIN_STD); print('Class weights:',CLASS_WEIGHTS); print('Loss: 50% weighted CE + 50% foreground Dice'); print('Optimizer: Adam'); print('Warmup LR:',WARMUP_LR); print('Fine-tune LR:',FINETUNE_LR); print('Max epochs:',MAX_EPOCHS); print('Early stopping patience:',EARLY_STOPPING_PATIENCE); print('TEST DATA LOADED: NO'); print('='*72)
    train_dataset=MTDataset(TRAIN_SCAN_DIR,TRAIN_MASK_DIR,True); val_dataset=MTDataset(VAL_SCAN_DIR,VAL_MASK_DIR,False)
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=False,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=False,num_workers=0)
    print(f'\nTraining samples: {len(train_dataset)}'); print(f'Validation samples: {len(val_dataset)}'); print('Test samples used: 0')
    model=build_hybrid_model().to(DEVICE)
    sample_scan,sample_mask=train_dataset[0]; model.eval()
    with torch.no_grad(): sample_logits=model(sample_scan.unsqueeze(0).to(DEVICE))
    print('\nMODEL SHAPE CHECK'); print('-'*72); print('Input:',tuple(sample_scan.unsqueeze(0).shape)); print('Output:',tuple(sample_logits.shape)); print('Mask:',tuple(sample_mask.shape))
    if sample_logits.shape[1]!=NUM_CLASSES: raise RuntimeError('Model output is not 3-class.')
    if tuple(sample_logits.shape[2:])!=tuple(sample_mask.shape): raise RuntimeError('Output spatial dimensions do not match mask.')
    print('3-CLASS OUTPUT CHECK: PASS')
    class_weights=torch.tensor(CLASS_WEIGHTS,dtype=torch.float32,device=DEVICE); ce_loss=nn.CrossEntropyLoss(weight=class_weights); dice_loss=ForegroundDiceLoss()
    csv_fields=['epoch','phase','learning_rate','seconds','train_loss','train_ce','train_dice_loss','val_loss','val_ce','val_dice_loss','mt_dice','mt_iou','mt_precision','mt_recall','air_dice','air_iou','air_precision','air_recall','foreground_mean_dice']
    with open(HISTORY_PATH,'w',newline='') as f: csv.DictWriter(f,fieldnames=csv_fields).writeheader()
    best_mt_dice=-1.0; best_foreground_dice=-1.0; no_improve=0; optimizer=None; scheduler=None; current_phase=None; last_completed_epoch=0; last_train_metrics={}; last_val_metrics={}
    try:
        for epoch in range(1,MAX_EPOCHS+1):
            epoch_start=time.time(); desired_phase='head_warmup' if epoch<=WARMUP_EPOCHS else 'full_finetune'
            if desired_phase!=current_phase:
                current_phase=desired_phase; print('\n'+'='*72); print('TRAINING PHASE:',current_phase); print('='*72)
                if current_phase=='head_warmup':
                    freeze_backbone_train_head_only(model); optimizer=torch.optim.Adam(model.conv_out.parameters(),lr=WARMUP_LR); scheduler=None; print('Backbone frozen; 3-class head trainable.')
                else:
                    unfreeze_entire_model(model); optimizer=torch.optim.Adam(model.parameters(),lr=FINETUNE_LR); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=SCHEDULER_FACTOR,patience=SCHEDULER_PATIENCE); print('Entire healthy-pretrained U-Net is now trainable.')
            train_metrics=train_one_epoch(model,train_loader,optimizer,ce_loss,dice_loss,current_phase=='head_warmup',epoch)
            val_metrics=validate_one_epoch(model,val_loader,ce_loss,dice_loss)
            if scheduler is not None: scheduler.step(val_metrics['loss'])
            lr=optimizer.param_groups[0]['lr']; seconds=time.time()-epoch_start
            print(f'\nEpoch {epoch:02d}/{MAX_EPOCHS}'); print('Phase:',current_phase)
            print(f"TRAIN loss={train_metrics['loss']:.4f} CE={train_metrics['ce']:.4f} DiceLoss={train_metrics['dice_loss']:.4f}")
            print(f"VAL   loss={val_metrics['loss']:.4f} CE={val_metrics['ce']:.4f} DiceLoss={val_metrics['dice_loss']:.4f}")
            print(f"MT    Dice={val_metrics['mt_dice']:.4f} IoU={val_metrics['mt_iou']:.4f} Precision={val_metrics['mt_precision']:.4f} Recall={val_metrics['mt_recall']:.4f}")
            print(f"AIR   Dice={val_metrics['air_dice']:.4f} IoU={val_metrics['air_iou']:.4f} Precision={val_metrics['air_precision']:.4f} Recall={val_metrics['air_recall']:.4f}")
            print(f"Mean foreground Dice={val_metrics['foreground_mean_dice']:.4f}"); print(f'Learning rate={lr:.7f}'); print(f'Epoch time={seconds:.1f} sec')
            mt_improved=val_metrics['mt_dice']>best_mt_dice+MIN_MT_IMPROVEMENT
            if mt_improved:
                best_mt_dice=val_metrics['mt_dice'];
                if current_phase=='full_finetune': no_improve=0
                checkpoint=make_checkpoint(model,optimizer,scheduler,epoch,current_phase,train_metrics,val_metrics,best_mt_dice,max(best_foreground_dice,val_metrics['foreground_mean_dice']))
                torch.save(checkpoint,BEST_MT_PATH); print('\n*** NEW BEST MT MODEL ***'); print(f'MT Dice = {best_mt_dice:.4f}'); print('Saved:',BEST_MT_PATH)
            elif current_phase=='full_finetune': no_improve+=1
            if val_metrics['foreground_mean_dice']>best_foreground_dice:
                best_foreground_dice=val_metrics['foreground_mean_dice']; checkpoint=make_checkpoint(model,optimizer,scheduler,epoch,current_phase,train_metrics,val_metrics,best_mt_dice,best_foreground_dice); torch.save(checkpoint,BEST_FOREGROUND_PATH); print('\n*** NEW BEST FOREGROUND MODEL ***'); print(f'Mean foreground Dice = {best_foreground_dice:.4f}')
            checkpoint=make_checkpoint(model,optimizer,scheduler,epoch,current_phase,train_metrics,val_metrics,best_mt_dice,best_foreground_dice); torch.save(checkpoint,LATEST_PATH)
            with open(HISTORY_PATH,'a',newline='') as f:
                csv.DictWriter(f,fieldnames=csv_fields).writerow({'epoch':epoch,'phase':current_phase,'learning_rate':lr,'seconds':seconds,'train_loss':train_metrics['loss'],'train_ce':train_metrics['ce'],'train_dice_loss':train_metrics['dice_loss'],'val_loss':val_metrics['loss'],'val_ce':val_metrics['ce'],'val_dice_loss':val_metrics['dice_loss'],'mt_dice':val_metrics['mt_dice'],'mt_iou':val_metrics['mt_iou'],'mt_precision':val_metrics['mt_precision'],'mt_recall':val_metrics['mt_recall'],'air_dice':val_metrics['air_dice'],'air_iou':val_metrics['air_iou'],'air_precision':val_metrics['air_precision'],'air_recall':val_metrics['air_recall'],'foreground_mean_dice':val_metrics['foreground_mean_dice']})
            last_completed_epoch=epoch; last_train_metrics=train_metrics; last_val_metrics=val_metrics
            print(f'\nBest validation MT Dice: {best_mt_dice:.4f}'); print(f'Best mean foreground Dice: {best_foreground_dice:.4f}')
            if current_phase=='full_finetune': print('Full-finetune epochs without MT improvement:',no_improve)
            print('='*72)
            if current_phase=='full_finetune' and no_improve>=EARLY_STOPPING_PATIENCE:
                print(f'\nEARLY STOPPING: no meaningful MT Dice improvement for {EARLY_STOPPING_PATIENCE} full-finetune epochs.'); break
            if DEVICE.type=='cuda': torch.cuda.empty_cache()
            elif DEVICE.type=='mps' and hasattr(torch.mps,'empty_cache'): torch.mps.empty_cache()
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user.')
        interrupted_checkpoint=make_checkpoint(model,optimizer,scheduler,last_completed_epoch,current_phase,last_train_metrics,last_val_metrics,best_mt_dice,best_foreground_dice); torch.save(interrupted_checkpoint,INTERRUPTED_PATH); print('Saved interrupted state:',INTERRUPTED_PATH)
    print('\n'+'='*72); print('TRAINING FINISHED'); print('='*72); print('Best MT checkpoint:',BEST_MT_PATH); print('Best foreground checkpoint:',BEST_FOREGROUND_PATH); print('Latest checkpoint:',LATEST_PATH); print('History CSV:',HISTORY_PATH); print(f'Best validation MT Dice: {best_mt_dice:.4f}'); print(f'Best validation foreground Dice: {best_foreground_dice:.4f}'); print('MT TEST SET WAS NEVER LOADED.'); print('='*72)

if __name__=='__main__': main()
