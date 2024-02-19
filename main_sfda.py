#/usr/bin/env python3.6
import math, pdb
import re
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from operator import itemgetter
from shutil import copytree, rmtree
import typing
from typing import Any, Callable, List, Tuple
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from dice3d import dice3d
from networks import weights_init
from dataloader import get_loaders
from utils import map_, save_dict_to_file
from utils import dice_coef, dice_batch, save_images,save_images_p,save_be_images, tqdm_, save_images_ent
from utils import probs2one_hot, probs2class, mask_resize, resize, haussdorf
from utils import exp_lr_scheduler
import datetime
from itertools import cycle
import os
from time import sleep, time
from bounds import CheckBounds
import matplotlib.pyplot as plt
from itertools import chain
import platform
from termcolor import colored, cprint
from copy import deepcopy;
import torch
import threading
from pathlib import Path
import nibabel as nib

class SaveThread:
    
    def __init__(self):
        self.lock = threading.Lock()

    def save_nii(self, img, filename):
        # Create a new thread for saving the NIfTI file
        thread = threading.Thread(target=self._save, args=(img, filename))
        thread.start()

    def _save(self, img, filename):
        # Acquire the lock
        with self.lock:
            # Make a copy of the image
            img_copy = deepcopy(img)

            # Save the NIfTI file
            nib.save(img_copy, filename)
            # print(f'Saved NIfTI file to {filename}')

def save_images_as_stacks(args, target_gt,  n_slices_per_out, st,
                          out_stack_image, out_stack_gt, out_stack_pred, out_stack_idx, out_stack_posZ,
                          pred_probs, target_data, target_image,
                          filenames_target, savedir, mode, epc, current_name, affine = np.diag(((0.5, 0.5, 0.5, 1)))):
    
    modality = "ct" if target_data[0][0].find("ctslice")>=0 else "mr"

    n_slices_thisbatch = target_image.shape[0]
    out_stack_image[..., out_stack_posZ:out_stack_posZ+n_slices_thisbatch] = np.transpose(target_image.cpu().numpy()[:,0,...], (1,2,0)) * 10000
    out_stack_pred [..., out_stack_posZ:out_stack_posZ+n_slices_thisbatch, :] = np.transpose(pred_probs.cpu().detach().numpy(), (2,3,0, 1))
    out_stack_gt   [..., out_stack_posZ:out_stack_posZ+n_slices_thisbatch] = np.transpose(np.argmax(target_gt.cpu().numpy(), axis=1), (1,2,0))

    assert out_stack_posZ % args.batch_size == 0
    
    if out_stack_posZ + args.batch_size < n_slices_per_out:
      out_stack_posZ = out_stack_posZ + args.batch_size
    else:
      if args.infdata =="all":
        out_stack_path_prefix = os.path.join(savedir, modality + "-" +  mode, current_name + "-" + str(n_slices_per_out * out_stack_idx))
      else:
        model_name = args.model_weights.split("/")[-1].replace(".pkl", "")
        out_stack_path_prefix = os.path.join(savedir, ( "inf" if args.mode=="makeim" else f"iter{epc:03d}" + "-" + model_name) +"-stack" , modality + "-" +  mode, "stack-" + str(n_slices_per_out * out_stack_idx))

      os.makedirs(os.path.dirname(out_stack_path_prefix), exist_ok=True)
      st.save_nii(nib.Nifti1Image( np.argmax(out_stack_pred [..., 0:out_stack_posZ+n_slices_thisbatch], axis=3).astype(np.uint8), affine), out_stack_path_prefix+"-pred.nii.gz")

      if epc==0 or mode=="train":
          # images and ground truth are saved only once on evaluation or testing
          # On training, images and ground-truths are different at each epoch due to shuffling and augmentation
          st.save_nii(nib.Nifti1Image(out_stack_image[..., 0:out_stack_posZ+n_slices_thisbatch], affine), out_stack_path_prefix+"-image.nii.gz")
          st.save_nii(nib.Nifti1Image(out_stack_gt   [..., 0:out_stack_posZ+n_slices_thisbatch], affine), out_stack_path_prefix+"-gt.nii.gz")
          
      else:
        source_path_prefix = os.path.join(savedir, ( "inf" if args.mode=="makeim" else f"iter{epc:03d}") + "-stack" , modality + "-" +  mode, "stack-" + str(n_slices_per_out * out_stack_idx))
        for suffix in (("-image.nii.gz", "-gt.nii.gz")):
          if os.path.islink(out_stack_path_prefix + suffix):
              os.unlink(out_stack_path_prefix + suffix)
          elif os.path.exists(out_stack_path_prefix + suffix):
              os.remove(out_stack_path_prefix + suffix)
          os.symlink(os.path.relpath(source_path_prefix + suffix, start=os.path.dirname(out_stack_path_prefix)), out_stack_path_prefix + suffix)

      out_stack_posZ = 0
      out_stack_idx += 1
    return out_stack_posZ, out_stack_idx


def save_model_with_info(model, model_path, optimizer, epoch, loss_history=None, verbose=True):

    # save model
    torch.save(model, model_path)

    # save training state
    state_path = str(model_path).replace(".pkl", "_state.pth")
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
    }, state_path)

    if verbose:
      print(f"Model saved to {model_path}, and state saved to {os.path.basename(state_path)}")


def setup(args, n_class, dtype) -> Tuple[Any, Any, Any, List[Callable], List[float],List[Callable], List[float], Callable]:
    print(">>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    if cpu:
        print("WARNING CUDA NOT AVAILABLE")
    device = torch.device("cpu") if cpu else torch.device("cuda")
    n_epoch = args.n_epoch

    if args.model_weights:
        net = torch.load(args.model_weights, map_location='cpu') if cpu else  torch.load(args.model_weights)

        # Check if the corresponding state file exists
        state_file_path = args.model_weights.replace(".pkl", "_state.pth")
        if os.path.exists(state_file_path):
            state = torch.load(state_file_path, map_location=device if cpu else None)
            print(f"{state_file_path} found and loaded.")
        else:
            state = None
        
        # Fixation of some weights on adaptation
        if args.network == "monai" and args.fixlayers:
          fixed_layers = ()

          if args.monai == "unetpp":
            fixed_layers = ('upcat_0_4', 'upcat_1_3', 'upcat_2_2', 'upcat_3_1', 'final_conv_0_4', 'final_conv_1_3', 'final_conv_2_2', 'final_conv_3_1')

          cprint("Fixed layers: ", "red", end=" ")
          for name, param in net.named_parameters():
              if np.sum([int(name.find(x)>0) for x in fixed_layers]) > 0:
                  print(name, end=", ")
                  param.requires_grad = False
          print()
        
    else:
        state = None
        net_class = getattr(__import__('networks'), args.network)

        if args.network == "monai":
          cprint("MONAI's network: " +args.monai, "green")
          net = net_class(1, n_class, network=args.monai).type(dtype).to(device)
        else:
          net = net_class(1, n_class).type(dtype).to(device)
          net.apply(weights_init)
    net.to(device)
    if args.saveim:
        print("WARNING SAVING MASKS at each epc")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999),weight_decay=args.weight_decay)
    if args.adamw:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999))
    
    if state is not None and args.resume==True:
      optimizer.load_state_dict(state['optimizer_state_dict'])
      print("Optimizer state loaded.")
      
      start_epoch = state['epoch'] + 1
      print("Training will start from epoch", start_epoch)

      loss_history = state['loss_history'] # currently not used
    
    else:
      start_epoch = 0
      loss_history = None

    print(args.target_losses)
    losses = eval(args.target_losses)
    loss_fns: List[Callable] = []
    for loss_name, loss_params, _, bounds_params, fn, _ in losses:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_params, dtype=dtype, fn=fn))
        print("bounds_params", bounds_params)
        if bounds_params!=None:
            bool_predexist = CheckBounds(**bounds_params)
            print(bool_predexist,"size predictor")
            if not bool_predexist:
                n_epoch = 0

    loss_weights = map_(itemgetter(5), losses)

    if args.scheduler:
        scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))
    else:
        scheduler = ''

    return net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, start_epoch, loss_history


def do_epoch(args, mode: str, net: Any, device: Any, epc: int,
             loss_fns: List[Callable], loss_weights: List[float],
              new_w:int, C: int, metric_axis:List[int], savedir: str = "",
             optimizer: Any = None, target_loader: Any = None, best_dice3d_val:Any=None):

    assert mode in ["train", "val"]
    L: int = len(loss_fns)

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_it_t, total_images_t = len(target_loader), len(target_loader.dataset)
    total_iteration = total_it_t
    total_images = total_images_t

    if args.debug:
        total_iteration = 10
    pho=1
    dtype = eval(args.dtype)

    all_dices: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_gt_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes2: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_inter_card: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_card_gt: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_card_pred: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_gt = []
    all_pred = []
    if args.do_hd: 
        #all_gt: Tensor = torch.zeros((total_images, 256, 256), dtype=dtype)
        all_gt: Tensor = torch.zeros((total_images, 384, 384), dtype=dtype)
        #all_pred: Tensor = torch.zeros((total_images, 256, 256), dtype=dtype)
        all_pred: Tensor = torch.zeros((total_images, 384, 384), dtype=dtype)
    loss_log: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_cons: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_se: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_tot: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    posim_log: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_grp: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_pnames = np.zeros([total_images]).astype('U256') 
    dice_3d_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    dice_3d_sd_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    hd95_3d_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    hd95_3d_sd_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    tq_iter = tqdm_(enumerate(target_loader), total=total_iteration, desc=desc)
    done: int = 0
    n_warmup = args.n_warmup
    mult_lw = [pho ** (epc - n_warmup + 1)] * len(loss_weights)
    mult_lw[0] = 1
    loss_weights = [a * b for a, b in zip(loss_weights, mult_lw)]
    losses_vec, source_vec, target_vec, baseline_target_vec = [], [], [], []
    pen_count = 0

    if total_images==0:
        print("total_images is zero!")
        pdb.set_trace()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count_losses = 0

        # For nifti outputs of patches
        def find_smallest_multiple(number, divisor):
          return number + divisor - (number % divisor) if number % divisor != 0 else number
        n_slices_per_out = find_smallest_multiple(256, args.batch_size)

        st = SaveThread()
        out_stack_image = np.zeros((256,  256, n_slices_per_out), dtype=np.int16)
        out_stack_pred  = np.zeros((256,  256, n_slices_per_out, args.n_class), dtype=np.float32)
        out_stack_gt    = np.zeros((256,  256, n_slices_per_out), dtype=np.uint8)
        out_stack_posZ  = 0 
        out_stack_idx   = 0

        for j, target_data in tq_iter:
            
            # Preparation for nii outputs
            tmp_name = ""
            start_time = time()
            current_name = target_data[0][0].split("_")[0]
            name_changed = False
            if args.infdata == "all" and j > 0:
                if tmp_name != current_name:
                    name_changed = True
                    out_stack_image.fill(0)
                    out_stack_pred.fill(0)
                    out_stack_gt.fill(0)
                    out_stack_posZ = 0
                    out_stack_idx = 0
            tmp_name = deepcopy(current_name)

            # Actual process
            target_data[1:] = [e.to(device) for e in target_data[1:]]  # Move all tensors to device
            filenames_target, target_image, target_gt = target_data[:3]
            labels = target_data[3:3+L]
            bounds = target_data[3+L:]
            filenames_target = [f.split('.nii')[0] for f in filenames_target]
            assert len(labels) == len(bounds), len(bounds)
            B = len(target_image)

            # Reset gradients
            if optimizer:
                #adjust_learning_rate(optimizer, 1, args.l_rate, args.power)
                optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(mode == "train"):
                pred_logits: Tensor = net(target_image)
                # pred_probs: Tensor = F.softmax(pred_logits, dim=1)
                if isinstance(pred_logits, list):
                    pred_probs = [F.softmax(logits, dim=1) for logits in pred_logits][0]
                else:
                    pred_probs = F.softmax(pred_logits, dim=1)
                
                predicted_mask: Tensor = probs2one_hot(pred_probs)  # Used only for dice computation
            assert len(bounds) == len(loss_fns) == len(loss_weights)
            if epc < n_warmup:
                loss_weights = [0]*len(loss_weights)
            loss: Tensor = torch.zeros(1, requires_grad=True).to(device)
            loss_vec = []
            loss_kw = []
            
            for loss_fn,label, w, bound in zip(loss_fns,labels, loss_weights, bounds):
                if w > 0:
                    if eval(args.target_losses)[0][0]=="EntKLProp": 
                        loss_1, loss_cons_prior,est_prop =  loss_fn(pred_probs, label, bound)
                        loss = loss_1 + loss_cons_prior 
                    else:
                        loss =  loss_fn(pred_probs, label, bound)
                        loss = w*loss
                        loss_1 = loss
                    loss_kw.append(loss_1.detach())

           # Backward
            if optimizer:
                loss.backward()
                optimizer.step()
                
            # Compute and log metrics
            dices, inter_card, card_gt, card_pred = dice_coef(predicted_mask.detach(), target_gt.detach())
            assert dices.shape == (B, C), (dices.shape, B, C)
            sm_slice = slice(done, done + B)  # Values only for current batch
            all_dices[sm_slice, ...] = dices
            if eval(args.target_losses)[0][0] in ["EntKLProp","WeightedEntKLProp","EntKLProp2","CEKLProp2"]:
                all_sizes[sm_slice, ...] = torch.round(est_prop.detach()*target_image.shape[2]*target_image.shape[3])
            all_sizes2[sm_slice, ...] = torch.sum(predicted_mask,dim=(2,3)) 
            all_gt_sizes[sm_slice, ...] = torch.sum(target_gt,dim=(2,3)) 
            # # for 3D dice
            '''
            if 'ctslice' in args.grp_regex:
                try:
                  all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('ctslice',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
                except:
                    pdb.set_trace()
            elif 'mrslice' in args.grp_regex:
                all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('mrslice',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
            '''
            
            if 'slice' in args.grp_regex:
                all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('slice',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
            elif 'Case' in args.grp_regex:
                all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('Case',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
            else:
                all_grp[sm_slice, ...] = int(re.split('_', filenames_target[0])[1]) * torch.ones([1, C])
            all_pnames[sm_slice] = filenames_target
            all_inter_card[sm_slice, ...] = inter_card
            all_card_gt[sm_slice, ...] = card_gt
            all_card_pred[sm_slice, ...] = card_pred
            if args.do_hd:
                all_pred[sm_slice, ...] = probs2class(predicted_mask[:,:,:,:]).cpu().detach()
                all_gt[sm_slice, ...] = probs2class(target_gt).detach()
            loss_se[sm_slice] = loss_kw[0]
            if len(loss_kw)>1:
            	loss_cons[sm_slice] = loss_kw[1]
            	loss_tot[sm_slice] = loss_kw[1]+loss_kw[0]
            else:
            	loss_cons[sm_slice] = 0
            	loss_tot[sm_slice] = loss_kw[0]
                
            # Save images as stacks
            if savedir and (args.saveim or
                            (mode=="val"   and (epc == 0 or (epc+1) % 5 == 0) or
                            (mode=="train" and (epc == 0 or (epc+1) % 5 == 0) and out_stack_idx == 0 ))):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.simplefilter("ignore") 
                    predicted_class: Tensor = probs2class(pred_probs)
                    # save_images(predicted_class, filenames_target, savedir, mode, epc, False)
                    out_stack_posZ, out_stack_idx = save_images_as_stacks(args, target_gt,  n_slices_per_out, st, out_stack_image, out_stack_gt, out_stack_pred, out_stack_idx, out_stack_posZ, pred_probs, target_data, target_image, filenames_target, savedir, mode, epc, current_name)

                    if args.entmap:
                        ent_map = torch.einsum("bcwh,bcwh->bwh", [-pred_probs, (pred_probs+1e-10).log()])
                        save_images_ent(ent_map, filenames_target, savedir,'ent_map', epc)


            # Logging
            big_slice = slice(0, done + B)  # Value for current and previous batches
            stat_dict = {**{f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis},
                         **{f"SZ{n}": all_sizes[big_slice, n].mean() for n in metric_axis},
                         **({f"DSC_source{n}": all_dices_s[big_slice, n].mean() for n in metric_axis}
                           if args.source_metrics else {})}

            size_dict = {**{f"SZ{n}": all_sizes[big_slice, n].mean() for n in metric_axis}}
            nice_dict = {k: f"{v:.4f}" for (k, v) in stat_dict.items()}
            done += B
            tq_iter.set_postfix(nice_dict)

            

    if args.dice_3d and (mode == 'val'):
        dice_3d_log, dice_3d_sd_log, asd_3d_log, asd_3d_sd_log, hd_3d_log, hd_3d_sd_log = dice3d(all_grp, all_inter_card, all_card_gt, all_card_pred,all_pred,all_gt,all_pnames,metric_axis,args.pprint,args.do_hd, best_dice3d_val)
          
    # Compute and log metrics for classes specified in metric_axis ("indices" as Tensor)
    indices = torch.tensor(metric_axis,device=device)
    dice_2d = torch.index_select(all_dices, 1, indices).mean().cpu().numpy()
    target_vec = [ dice_3d_log, dice_3d_sd_log,hd95_3d_log,hd95_3d_sd_log,dice_2d]
    size_mean = torch.index_select(all_sizes2, 1, indices).mean(dim=0).cpu().numpy()
    size_gt_mean = torch.index_select(all_gt_sizes, 1, indices).mean(dim=0).cpu().numpy()
    mask_pos = torch.index_select(all_sizes2, 1, indices)!=0
    gt_pos = torch.index_select(all_gt_sizes, 1, indices)!=0
    size_mean_pos = torch.index_select(all_sizes2, 1, indices).sum(dim=0).cpu().numpy()/mask_pos.sum(dim=0).cpu().numpy()
    gt_size_mean_pos = torch.index_select(all_gt_sizes, 1, indices).sum(dim=0).cpu().numpy()/gt_pos.sum(dim=0).cpu().numpy()
    size_mean2 = torch.index_select(all_sizes2, 1, indices).mean(dim=0).cpu().numpy()

    try:
      losses_vec = [np.nanmean(loss_se.cpu().numpy()),np.nanmean(loss_cons.cpu().numpy()), np.nanmean(loss_tot.cpu().numpy()), np.int32(np.nanmean(size_mean)), np.int32(np.nanmean(size_mean_pos)), np.int32(np.nanmean(size_gt_mean)), np.int32(np.nanmean(gt_size_mean_pos)) ]
    except Exception as exc:
      import traceback; print(traceback.format_exc())
      losses_vec = [0,0,0,0,0,0,0]
      pdb.set_trace()
    if not epc%10:
        try:
          df_t # check existence
        except:
          df_t = pd.DataFrame(columns=["val_ids", "proposal_size"])

        df_t = pd.DataFrame({
           "val_ids":all_pnames,
           "proposal_size":all_sizes2.cpu().numpy().tolist()})
        df_t.to_csv(Path(savedir,mode+str(epc)+"sizes.csv"), float_format="%.4f", index_label="epoch")

    return losses_vec, target_vec,source_vec

def run(args: argparse.Namespace) -> None:
    
    d = vars(args)
    d['time'] = str(datetime.datetime.now())
    d['server']=platform.node()
    save_dict_to_file(d,args.workdir)
    temperature: float = 0.1
    n_class: int = args.n_class
    metric_axis: List = args.metric_axis
    lr: float = args.l_rate
    dtype = eval(args.dtype)
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch

    net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch, start_epoch, loss_history = setup(args, n_class, dtype)
    # currently loss_history is not used

    shuffle = True
    print(args.target_folders)
    target_loader, target_loader_val = get_loaders(args, args.target_dataset,args.target_folders,
                                           args.batch_size, n_class,
                                           args.debug, args.in_memory, dtype, shuffle, "target", args.val_target_folders)

    print("metric axis",metric_axis)
    best_dice_pos, best_dice, best_hd3d_dice = np.zeros(1), np.zeros(1), np.zeros(1)
    best_3d_dice = best_2d_dice = 0
    print("Results will be saved in ", savedir)


    print(">>> Starting the training")
    for i in range(start_epoch, n_epoch):

       if args.mode =="makeim":
            with torch.no_grad():
                val_losses_vec, val_target_vec,val_source_vec = do_epoch(
                    args, "val", net, device, i, loss_fns,
                    loss_weights,
                    args.resize,
                    n_class,metric_axis,
                    savedir=savedir,
                    target_loader=target_loader_val, best_dice3d_val=best_3d_dice)
                
                tra_losses_vec = val_losses_vec
                tra_target_vec = val_target_vec
                tra_source_vec = val_source_vec
       else:
            if True: # Set False to debug validation phase
              tra_losses_vec, tra_target_vec,tra_source_vec    = do_epoch(
                  args, "train", net, device,
                  i, loss_fns,
                  loss_weights,
                  args.resize,
                  n_class, metric_axis,
                  savedir=savedir,
                  optimizer=optimizer,
                  target_loader=target_loader, best_dice3d_val=best_3d_dice)
            with torch.no_grad():
                val_losses_vec, val_target_vec,val_source_vec = do_epoch(
                    args, "val", net, device,
                    i, loss_fns,
                    loss_weights,
                    args.resize,
                    n_class,metric_axis,
                    savedir=savedir,
                    target_loader=target_loader_val, best_dice3d_val=best_3d_dice)
                
       current_val_target_2d_dice = val_target_vec[4]
       current_val_target_3d_dice = val_target_vec[0]
       if args.dice_3d:
           if current_val_target_3d_dice > best_3d_dice:
               best_epoch = i
               best_3d_dice = current_val_target_3d_dice
               with open(Path(savedir, "3dbestepoch.txt"), 'w') as f:
                   f.write(str(i)+','+str(best_3d_dice))
               best_folder_3d = Path(savedir, "best_epoch_3d")
               if best_folder_3d.exists():
                    rmtree(best_folder_3d)
               if args.saveim:
                    copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_3d))
           # torch.save(net, Path(savedir, "best_3d.pkl"))
           save_model_with_info(net, Path(savedir, "best_3d.pkl"), optimizer, i, [tra_losses_vec, val_losses_vec])

       if i == 0 or (i+1) % 5 == 0:
            print("epoch",str(i),savedir,'best 3d dice',best_3d_dice)
            # torch.save(net, Path(savedir, "epoch_"+str(i)+".pkl"))
            save_model_with_info(net, Path(savedir, "epoch_"+str(i+1)+".pkl"), optimizer, i, [tra_losses_vec, val_losses_vec])
            
       if i == n_epoch - 1:
            with open(Path(savedir, "last_epoch.txt"), 'w') as f:
                f.write(str(i))
            last_folder = Path(savedir, "last_epoch")
            if last_folder.exists():
                try:
                    rmtree(last_folder)
                except Exception as exc:
                    print(exc)
            if args.saveim:
                try:
                  copytree(Path(savedir, f"iter{i:03d}"), Path(last_folder))
                except Exception as exc:
                    print(exc)
            # torch.save(net, Path(savedir, "last.pkl"))
            save_model_with_info(net, Path(savedir, "last.pkl"), optimizer, i, [tra_losses_vec, val_losses_vec])

        # remove images from iteration
       if args.saveim:
           # rmtree(Path(savedir, f"iter{i:03d}"))
           pass

       try:
        df_t_tmp = pd.DataFrame({
              "epoch":i,
              "tra_loss_s":[tra_losses_vec[0]],
              "tra_loss_cons":[tra_losses_vec[1]],
              "tra_loss_tot":[tra_losses_vec[2]],
              "tra_size_mean":[tra_losses_vec[3]],
              "tra_size_mean_pos":[tra_losses_vec[4]],
              "val_loss_s":[val_losses_vec[0]],
              "val_loss_cons":[val_losses_vec[1]],
              "val_loss_tot":[val_losses_vec[2]],
              "val_size_mean":[val_losses_vec[3]],
              "val_size_mean_pos":[val_losses_vec[4]],
              "val_gt_size_mean":[val_losses_vec[5]],
              "val_gt_size_mean_pos":[val_losses_vec[6]],
              'tra_dice': [tra_target_vec[4]],
              'val_dice': [val_target_vec[4]],
              "val_dice_3d_sd": [val_target_vec[1].cpu().numpy()],
              "val_dice_3d": [val_target_vec[0].cpu().numpy()]})
       except Exception as exc:
           print(exc)
           pdb.set_trace()


       if i == start_epoch:
            df_t = df_t_tmp
       else:
            df_t = df_t.append(df_t_tmp)

       df_t.to_csv(Path(savedir, "_".join((args.target_folders.split("'")[1],"target", args.csv))), float_format="%.4f", index=False)

       if args.flr==False:
            exp_lr_scheduler(optimizer, i, args.lr_decay,args.lr_decay_epoch)
    print("Results saved in ", savedir, "best 3d dice",best_3d_dice)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--target_dataset', type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--target_losses", type=str, required=True,
                        help="List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--val_target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--monai", type=str, default="UNet", help="If --network is monai, specify the model name")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--mode", type=str, default="learn")
    parser.add_argument("--lin_aug_w", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--trainval", action="store_true")
    parser.add_argument("--valonly", action="store_true")
    parser.add_argument("--flr", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mix", type=bool, default=True) # dangerous
    parser.add_argument("--do_hd", action="store_true")
    parser.add_argument("--saveim", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--csv", type=str, default='metrics.csv')
    parser.add_argument("--source_metrics", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--dice_3d", action="store_true")
    parser.add_argument("--ontest", action="store_true")
    parser.add_argument("--fixlayers", action="store_true")
    parser.add_argument("--ontrain", action="store_true")
    parser.add_argument("--best_losses", action="store_true")
    parser.add_argument("--pprint", action="store_true")
    parser.add_argument("--entmap", action="store_true")
    parser.add_argument("--model_weights", type=str, default='')
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--pho", nargs='?', type=float, default=1,
                        help='augment')
    parser.add_argument("--n_warmup", type=int, default=0)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=0.7),
    parser.add_argument('--lr_decay_epoch', nargs='?', type=float, default=20),
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-5,
                        help='L2 regularisation of network weights')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--power",type=float, default=0.9)
    parser.add_argument("--metric_axis",type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")
    parser.add_argument("--infdata", type=str, default="train", choices=["train", "general", "all"], help="train: non-inference mode. general: training and validation dataset. all: images that can reconstruct original nii")
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    run(get_args())

