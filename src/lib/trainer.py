from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar
from tqdm import tqdm

from model.data_parallel import DataParallel
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import fusion_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process
from model.losses import DepthLoss
from utils.pointcloud import generate_pc_hm

import cv2
class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss(opt=opt)
    self.crit_reg = RegWeightedL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.opt = opt
    self.crit_dep = DepthLoss()

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    if 'dep_sec' in output and self.opt.sigmoid_dep_sec:
      output['dep_sec'] = 1. / (output['dep_sec'].sigmoid() + 1e-6) - 1.
    return output

  def forward(self, outputs, batch):
    opt = self.opt
    losses = {head: 0 for head in opt.heads}

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = self._sigmoid_output(output)

      if 'hm' in output:
        losses['hm'] += self.crit(
          output['hm'], batch['hm'], batch['ind'], 
          batch['mask'], batch['cat']) / opt.num_stacks
      
      if 'dep' in output:
        losses['dep'] += self.crit_dep(
          output['dep'], batch['dep'], batch['ind'], 
          batch['dep_mask'], batch['cat']) / opt.num_stacks

      regression_heads = [
        'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps', 
        'dim', 'amodel_offset', 'velocity']

      for head in regression_heads:
        if head in output:
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks
      
      if 'hm_hp' in output:
        losses['hm_hp'] += self.crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'], 
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output:
          losses['hp_offset'] += self.crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        
      if 'rot' in output:
        losses['rot'] += self.crit_rot(
          output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres']) / opt.num_stacks

      if 'nuscenes_att' in output:
        losses['nuscenes_att'] += self.crit_nuscenes_att(
          output['nuscenes_att'], batch['nuscenes_att_mask'],
          batch['ind'], batch['nuscenes_att']) / opt.num_stacks
      
      if 'dep_sec' in output:
        losses['dep_sec'] += self.crit_dep(
          output['dep_sec'], batch['dep'], batch['ind'], 
          batch['dep_mask'], batch['cat']) / opt.num_stacks
      
      if 'rot_sec' in output:
        losses['rot_sec'] += self.crit_rot(
          output['rot_sec'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres']) / opt.num_stacks

    losses['tot'] = 0
    for head in opt.heads:
      losses['tot'] += opt.weights[head] * losses[head]
    scalar_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    return losses['tot'], scalar_losses


class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss, opt):
    super(ModelWithLoss, self).__init__()
    self.opt = opt
    self.model = model
    self.loss = loss
  
  def forward(self, batch, phase):
    pc_dep = batch.get('pc_dep', None)
    pc_hm = batch.get('pc_hm', None)
    calib = batch['calib'].squeeze(0)

    ## run the first stage
    outputs = self.model(batch['image'], pc_hm=pc_hm, pc_dep=pc_dep, calib=calib)
    
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats


class Trainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss, opt)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
                      if l == 'tot' or opt.weights[l] > 0}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    #bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    pbar = tqdm(enumerate(data_loader), total=num_iters, desc=f'{opt.task}-{phase} epoch {epoch}')
    end = time.time()
    
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)  
      
      # run one iteration 
      output, loss, loss_stats = model_with_loss(batch, phase)
      
      # backpropagate and step optimizer
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      desc = f'{phase}-epoch[{epoch}]: [{iter_id}/{num_iters}]'
      #Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
      # epoch, iter_id, num_iters, phase=phase,
      # total=bar.elapsed_td, eta=bar.eta_td)
      tracked_losses = ['tot', 'hm', 'dep', 'velocity']
      postfix_dict = {}
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          #loss_stats[l].mean().item(), batch['image'].size(0))
          loss_stats[l], batch['image'].size(0))
        if l in tracked_losses:
          postfix_dict[l] = f'{avg_loss_stats[l].avg:.3f}'
          #desc += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        #Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      #Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
      #  '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      #desc += f'|Data {data_time.val:.3f}s({data_time.avg:.3f}s)|Net {batch_time.avg:.3f}s'
      pbar.set_description(desc)
      postfix_dict['time'] = f'{batch_time.avg:.3f}s'
      pbar.set_postfix(**postfix_dict)

      if opt.print_iter > 0: # If not using progress bar
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        #bar.next()
        pbar.update(1)
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id, dataset=data_loader.dataset)
      
      # generate detections for evaluation
      if (phase == 'val' and (opt.run_dataset_eval or opt.eval)):
        meta = batch['meta']
        dets = fusion_decode(output, K=opt.K, opt=opt)

        for k in dets:
          dets[k] = dets[k].detach().cpu().numpy()

        calib = meta['calib'].detach().numpy() if 'calib' in meta else None
        dets = generic_post_process(opt, dets, 
          meta['c'].cpu().numpy(), meta['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)

        # merge results
        result = []
        for i in range(len(dets[0])):
          if dets[0][i]['score'] > self.opt.out_thresh and all(dets[0][i]['dim'] > 0):
            result.append(dets[0][i])

        img_id = batch['meta']['img_id'].numpy().astype(np.int32)[0]
        results[img_id] = result
 
      del output, loss, loss_stats
    
    #bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    #ret['time'] = bar.elapsed_td.total_seconds() / 60.
    ret['time'] = pbar.format_dict['elapsed'] / 60.
    return ret, results


  def _get_losses(self, opt):
    loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
      'hp_offset', 'dep', 'dep_sec', 'dim', 'rot', 'rot_sec',
      'amodel_offset', 'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
    loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
    loss = GenericLoss(opt)
    return loss_states, loss


  def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
      output.update({'pre_hm': batch['pre_hm']})
    dets = fusion_decode(output, K=opt.K, opt=opt)
    for k in dets:
      dets[k] = dets[k].detach().cpu().numpy()
    dets_gt = batch['meta']['gt_det']
    for i in range(1):
      debugger = Debugger(opt=opt, dataset=dataset)
      img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm', trans=self.opt.hm_transparency)
      debugger.add_blend_img(img, gt, 'gt_hm', trans=self.opt.hm_transparency)
      
      debugger.add_img(img, img_id='img')
      
      # show point clouds
      if opt.pointcloud:
        pc_2d = batch['pc_2d'][i].detach().cpu().numpy()
        pc_3d = None
        pc_N = batch['pc_N'][i].detach().cpu().numpy()
        debugger.add_img(img, img_id='pc')
        debugger.add_pointcloud(pc_2d, pc_N, img_id='pc')
        
        if 'pc_hm' in opt.pc_feat_lvl:
          channel = opt.pc_feat_channels['pc_hm']
          pc_hm = debugger.gen_colormap(batch['pc_hm'][i][channel].unsqueeze(0).detach().cpu().numpy())
          debugger.add_blend_img(img, pc_hm, 'pc_hm', trans=self.opt.hm_transparency)
        if 'pc_dep' in opt.pc_feat_lvl:
          channel = opt.pc_feat_channels['pc_dep']
          pc_hm = batch['pc_hm'][i][channel].unsqueeze(0).detach().cpu().numpy()
          pc_dep = debugger.add_overlay_img(img, pc_hm, 'pc_dep')
          

      if 'pre_img' in batch:
        pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_pred')
        debugger.add_img(pre_img, 'pre_img_gt')
        if 'pre_hm' in batch:
          pre_hm = debugger.gen_colormap(
            batch['pre_hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm', trans=self.opt.hm_transparency)

      debugger.add_img(img, img_id='out_pred')
      if 'ltrb_amodal' in opt.heads:
        debugger.add_img(img, img_id='out_pred_amodal')
        debugger.add_img(img, img_id='out_gt_amodal')

      # Predictions
      for k in range(len(dets['scores'][i])):
        if dets['scores'][i, k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
            dets['scores'][i, k], img_id='out_pred')

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
              dets['scores'][i, k], img_id='out_pred_amodal')

          if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
            debugger.add_coco_hp(
              dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

      # Ground truth
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt['scores'][i])):
        if dets_gt['scores'][i][k] > opt.vis_thresh:
          if 'dep' in dets_gt.keys():
            dist = dets_gt['dep'][i][k]
            if len(dist)>1:
              dist = dist[0]
          else:
            dist = -1
          debugger.add_coco_bbox(
            dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
            dets_gt['scores'][i][k], img_id='out_gt', dist=dist)

          if 'ltrb_amodal' in opt.heads:
            debugger.add_coco_bbox(
              dets_gt['bboxes_amodal'][i, k] * opt.down_ratio, 
              dets_gt['clses'][i, k],
              dets_gt['scores'][i, k], img_id='out_gt_amodal')

          if 'hps' in opt.heads and \
            (int(dets['clses'][i, k]) == 0):
            debugger.add_coco_hp(
              dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

      if 'hm_hp' in opt.heads:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp', trans=self.opt.hm_transparency)
        debugger.add_blend_img(img, gt, 'gt_hmhp', trans=self.opt.hm_transparency)


      if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
        dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
        calib = batch['meta']['calib'].detach().numpy() \
                if 'calib' in batch['meta'] else None
        det_pred = generic_post_process(opt, dets, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)
        det_gt = generic_post_process(opt, dets_gt, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib, is_gt=True)

        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i],
          det_pred[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i], 
          det_gt[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_gt')
        
        pc_3d = None
        if opt.pointcloud:
          pc_3d=batch['pc_3d'].cpu().numpy()

        debugger.add_bird_views(det_pred[i], det_gt[i], vis_thresh=opt.vis_thresh, 
          img_id='bird_pred_gt', pc_3d=pc_3d, show_velocity=opt.show_velocity)
        debugger.add_bird_views([], det_gt[i], vis_thresh=opt.vis_thresh, 
          img_id='bird_gt', pc_3d=pc_3d, show_velocity=opt.show_velocity)

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
  
  def measure_time(self, phase, data_loader, num_iterations=100):
      """
      測量模型正向傳播和反向傳播所需時間
      
      Args:
          phase: 'train' 或 'val'，指定是訓練或評估階段
          data_loader: 數據載入器
          num_iterations: 用於測量的迭代次數
      
      Returns:
          dict: 包含時間測量結果的字典
      """
      model_with_loss = self.model_with_loss
      if phase == 'train':
          model_with_loss.train()
      else:
          if len(self.opt.gpus) > 1:
              model_with_loss = self.model_with_loss.module
          model_with_loss.eval()
          torch.cuda.empty_cache()
      
      # 儲存時間測量結果
      forward_times = []
      backward_times = []
      data_loading_times = []
      total_times = []
      
      # 使用數據迭代器
      data_iter = iter(data_loader)
      
      # 先預熱GPU
      print("預熱GPU...")
      for _ in range(min(10, len(data_loader))):
          try:
              batch = next(data_iter)
          except StopIteration:
              data_iter = iter(data_loader)
              batch = next(data_iter)
              
          for k in batch:
              if k != 'meta':
                  batch[k] = batch[k].to(device=self.opt.device, non_blocking=True)
          
          # 運行一次迭代
          with torch.no_grad():
              model_with_loss(batch, phase)
      
      torch.cuda.synchronize()
      print(f"開始測量{num_iterations}次迭代的時間...")
      
      # 進行測量
      for i in range(min(num_iterations, len(data_loader))):
          try:
              batch = next(data_iter)
          except StopIteration:
              data_iter = iter(data_loader)
              batch = next(data_iter)
          
          # 測量數據載入時間
          data_start = time.time()
          for k in batch:
              if k != 'meta':
                  batch[k] = batch[k].to(device=self.opt.device, non_blocking=True)
          torch.cuda.synchronize()
          data_loading_times.append(time.time() - data_start)
          
          # 測量正向傳播時間
          torch.cuda.synchronize()
          forward_start = time.time()
          output, loss, loss_stats = model_with_loss(batch, phase)
          torch.cuda.synchronize()
          forward_times.append(time.time() - forward_start)
          
          # 測量反向傳播時間（僅訓練階段）
          if phase == 'train':
              backward_start = time.time()
              self.optimizer.zero_grad()
              loss.mean().backward()
              torch.cuda.synchronize()
              backward_time = time.time() - backward_start
              backward_times.append(backward_time)
              
              # 不實際更新參數，避免影響模型
              # self.optimizer.step()
          
          # 總時間
          total_times.append(data_loading_times[-1] + forward_times[-1] + 
                          (backward_times[-1] if phase == 'train' else 0))
          
          # 實時顯示進度
          if (i + 1) % 10 == 0:
              print(f"已完成 {i+1}/{num_iterations} 次測量")
      
      # 計算統計數據
      data_mean = np.mean(data_loading_times)
      data_std = np.std(data_loading_times)
      forward_mean = np.mean(forward_times)
      forward_std = np.std(forward_times)
      
      results = {
          'data_loading': {
              'mean': data_mean,
              'std': data_std,
              'raw': data_loading_times
          },
          'forward': {
              'mean': forward_mean,
              'std': forward_std,
              'raw': forward_times
          }
      }
      
      if phase == 'train':
          backward_mean = np.mean(backward_times)
          backward_std = np.std(backward_times)
          results['backward'] = {
              'mean': backward_mean,
              'std': backward_std,
              'raw': backward_times
          }
          results['forward_backward_ratio'] = backward_mean / forward_mean
      
      results['total'] = {
          'mean': np.mean(total_times),
          'std': np.std(total_times),
          'raw': total_times
      }
      
      # 打印測量結果
      print("\n時間測量結果:")
      print(f"數據載入時間: {data_mean:.4f} ± {data_std:.4f} 秒")
      print(f"正向傳播時間: {forward_mean:.4f} ± {forward_std:.4f} 秒")
      
      if phase == 'train':
          print(f"反向傳播時間: {backward_mean:.4f} ± {backward_std:.4f} 秒")
          print(f"反向:正向比例: {backward_mean/forward_mean:.2f}")
      
      print(f"總時間: {np.mean(total_times):.4f} ± {np.std(total_times):.4f} 秒")
      print(f"批次大小: {batch['image'].shape[0]}")
      
      # 每個樣本的時間
      per_sample_forward = forward_mean / batch['image'].shape[0]
      print(f"每個樣本正向傳播時間: {per_sample_forward:.4f} 秒")
      
      if phase == 'train':
          per_sample_backward = backward_mean / batch['image'].shape[0]
          print(f"每個樣本反向傳播時間: {per_sample_backward:.4f} 秒")
          print(f"每個樣本總時間: {(forward_mean + backward_mean) / batch['image'].shape[0]:.4f} 秒")
      
      return results