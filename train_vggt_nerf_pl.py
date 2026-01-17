import torch
from opt import config_parser
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import imageio
from data import dataset_dict

# models
from models.models import *
from utils.renderer import *
from utils import *

# optimizer, scheduler, visualization
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss

class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.idx = 0

        self.loss = SL1Loss()
        self.learning_rate = args.lrate

        self.validation_step_outputs = [] # 新增：用于存储验证步的输出
        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']
        self.render_kwargs_train.pop('network_mvs')

        # ==========================================================
        # 🔥🔥🔥 核心修复：必须把 NeRF 网络显式赋值给 self 🔥🔥🔥
        # ==========================================================
        # 只有这样，Lightning 才会把它当成是 System 的一部分，保存进 checkpoint
        if 'network_fn' in self.render_kwargs_train:
            self.network_fn = self.render_kwargs_train['network_fn']
            print("   ✅ Registered 'network_fn' to MVSSystem (Will be saved in ckpt).")
            
        if 'network_fine' in self.render_kwargs_train:
            self.network_fine = self.render_kwargs_train['network_fine']
            print("   ✅ Registered 'network_fine' to MVSSystem (Will be saved in ckpt).")
        # ==========================================================
        self.render_kwargs_train['NDC_local'] = False

        self.eval_metric = [0.01,0.05, 0.1]


    def decode_batch(self, batch, idx=list(torch.arange(4))):

        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
        pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                    'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}

        return data_mvs, pose_ref

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std


    def forward(self):
        return

    def prepare_data(self):
        dataset = dataset_dict[self.args.dataset_name]
        train_dir, val_dir = self.args.datadir , self.args.datadir
        self.train_dataset = dataset(self.args, split='train')
        self.val_dataset   = dataset(self.args, split='val')#


    def configure_optimizers(self):
        eps = 1e-7
        self.optimizer = torch.optim.Adam(self.grad_vars, lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.num_epochs, eta_min=eps)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=16,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=8,
                          batch_size=1,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        if 'scan' in batch.keys():
            batch.pop('scan')
        log, loss = {},0
        data_mvs, pose_ref = self.decode_batch(batch)
        imgs, proj_mats = data_mvs['images'], data_mvs['proj_mats']
        near_fars, depths_h = data_mvs['near_fars'], data_mvs['depths_h']


        volume_feature, img_feat, depth_values = self.MVSNet(imgs[:, :3], proj_mats[:, :3], near_fars[0,0],pad=args.pad)
        imgs = self.unpreprocess(imgs)


        N_rays, N_samples = args.batch_size, args.N_samples
        c2ws, w2cs, intrinsics = pose_ref['c2ws'], pose_ref['w2cs'], pose_ref['intrinsics']
        rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters = \
            build_rays(imgs, depths_h, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=args.pad)


        rgb, disp, acc, depth_pred, alpha, ret = rendering(args, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs[:, :-1], img_feat=None,  **self.render_kwargs_train)


        if self.args.with_depth:
            mask = rays_depth > 0
            if self.args.with_depth_loss:
                loss += self.loss(depth_pred, rays_depth, mask)

            self.log(f'train/acc_l_{self.eval_metric[0]}mm', acc_threshold(depth_pred, rays_depth, mask,
                                                                      self.eval_metric[0]).mean(), prog_bar=False)
            self.log(f'train/acc_l_{self.eval_metric[1]}mm', acc_threshold(depth_pred, rays_depth, mask,
                                                                      self.eval_metric[1]).mean(), prog_bar=False)
            self.log(f'train/acc_l_{self.eval_metric[2]}mm', acc_threshold(depth_pred, rays_depth, mask,
                                                                      self.eval_metric[2]).mean(), prog_bar=False)

            abs_err = abs_error(depth_pred, rays_depth, mask).mean()
            self.log('train/abs_err', abs_err, prog_bar=True)

        ##################  rendering #####################
        img_loss = img2mse(rgb, target_s)
        loss = loss + img_loss
        if 'rgb0' in ret:
            img_loss_coarse = img2mse(ret['rgb0'], target_s)
            psnr = mse2psnr2(img_loss_coarse.item())
            self.log('train/PSNR_coarse', psnr.item(), prog_bar=True)
            loss = loss + img_loss_coarse


        if args.with_depth:
            psnr = mse2psnr(img2mse(rgb.cpu()[mask], target_s.cpu()[mask]))
            psnr_out = mse2psnr(img2mse(rgb.cpu()[~mask], target_s.cpu()[~mask]))
            self.log('train/PSNR_out', psnr_out.item(), prog_bar=True)
        else:
            psnr = mse2psnr2(img_loss.item())

        with torch.no_grad():
            self.log('train/loss', loss, prog_bar=True)
            self.log('train/img_mse_loss', img_loss.item(), prog_bar=False)
            self.log('train/PSNR', psnr.item(), prog_bar=True)

        if self.global_step % 20000==19999:
            self.save_ckpt(f'{self.global_step}')


        return  {'loss':loss}



    def validation_step(self, batch, batch_nb):
        if(batch_nb%5!=0):
            return
        if 'scan' in batch.keys():
            batch.pop('scan')

        log = {}
        data_mvs, pose_ref = self.decode_batch(batch)
        imgs, proj_mats = data_mvs['images'], data_mvs['proj_mats']
        near_fars, depths_h = pose_ref['near_fars'], data_mvs['depths_h']

        self.MVSNet.train()
        H, W = imgs.shape[-2:]
        H, W = int(H), int(W)


        ##################  rendering #####################
        keys = ['val_psnr', 'val_depth_loss_r', 'val_abs_err', 'mask_sum'] + [f'val_acc_{i}mm' for i in self.eval_metric]
        log = init_log(log, keys)
        with torch.no_grad():

            args.img_downscale = torch.rand((1,)) * 0.75 + 0.25  # for super resolution
            world_to_ref = pose_ref['w2cs'][0]
            tgt_to_world, intrinsic = pose_ref['c2ws'][-1], pose_ref['intrinsics'][-1]
            volume_feature, img_feat, _ = self.MVSNet(imgs[:, :3], proj_mats[:, :3], near_fars[0], pad=args.pad)
            imgs = self.unpreprocess(imgs)
            rgbs, depth_preds = [],[]
            for chunk_idx in range(H*W//args.chunk + int(H*W%args.chunk>0)):


                rays_pts, rays_dir, rays_NDC, depth_candidates, rays_o, ndc_parameters = \
                    build_rays_test(H, W, tgt_to_world, world_to_ref, intrinsic, near_fars, near_fars[-1], args.N_samples, pad=args.pad, chunk=args.chunk, idx=chunk_idx)


                # rendering
                rgb, disp, acc, depth_pred, density_ray, ret = rendering(args, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs[:, :-1], img_feat=None,  **self.render_kwargs_train)
                rgbs.append(rgb.cpu());depth_preds.append(depth_pred.cpu())

            imgs = imgs.cpu()
            rgb, depth_r = torch.clamp(torch.cat(rgbs).reshape(H, W, 3).permute(2,0,1),0,1), torch.cat(depth_preds).reshape(H, W)
            img_err_abs = (rgb - imgs[0,-1]).abs()

            if self.args.with_depth:
                depth_gt_render = depths_h[0, -1].cpu()
                mask = depth_gt_render > 0
                log['val_psnr'] = mse2psnr(torch.mean(img_err_abs[:,mask] ** 2))
            else:
                log['val_psnr'] = mse2psnr(torch.mean(img_err_abs**2))


            if self.args.with_depth:

                log['val_depth_loss_r'] = self.loss(depth_r, depth_gt_render, mask)

                minmax = [2.0,6.0]
                depth_gt_render_vis,_ = visualize_depth(depth_gt_render,minmax)
                depth_pred_r_, _ = visualize_depth(depth_r, minmax)
                depth_err_, _ = visualize_depth(torch.abs(depth_r-depth_gt_render)*5, minmax)
                img_vis = torch.stack((depth_gt_render_vis, depth_pred_r_, depth_err_))
                self.logger.experiment.add_images('val/depth_gt_pred_err', img_vis, self.global_step)

                log['val_abs_err'] = abs_error(depth_r, depth_gt_render, mask).sum()
                log[f'val_acc_{self.eval_metric[0]}mm'] = acc_threshold(depth_r, depth_gt_render, mask, self.eval_metric[0]).sum()
                log[f'val_acc_{self.eval_metric[1]}mm'] = acc_threshold(depth_r, depth_gt_render, mask, self.eval_metric[1]).sum()
                log[f'val_acc_{self.eval_metric[2]}mm'] = acc_threshold(depth_r, depth_gt_render, mask, self.eval_metric[2]).sum()
                log['mask_sum'] = mask.float().sum()
            else:
                minmax = [2.0, 6.0]
                depth_pred_r_, _ = visualize_depth(depth_r, minmax)
                self.logger.experiment.add_images('val/depth_gt_pred_err', depth_pred_r_[None], self.global_step)

            imgs = imgs[0]
            img_vis = torch.cat((imgs, torch.stack((rgb, img_err_abs.cpu()*5))), dim=0) # N 3 H W
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step)

            # os.makedirs(f'runs_new/{self.args.expname}/{self.args.expname}/',exist_ok=True)
            # 1. 图片处理 (保持你原来的代码一字不改，这样你就放心了)
            # 这行代码把 GT、RGB、Error 和 Depth 拼在一起
            img_vis = torch.cat((img_vis,depth_pred_r_[None]),dim=0).permute(2,0,3,1).reshape(img_vis.shape[2],-1,3).numpy()

            # 2. 路径处理 (只改这里！)
            # 创建带 Epoch 的文件夹: runs_new/实验名/validation_images/epoch_02/
            save_dir = f'runs_new/{self.args.expname}/validation_images/epoch_{self.current_epoch:02d}'
            os.makedirs(save_dir, exist_ok=True)

            # 3. 保存 (改文件名)
            # 用 batch_nb 命名 (000.png, 001.png...)
            filename = f'{save_dir}/{batch_nb:03d}.png'
            imageio.imwrite(filename, (img_vis*255).astype('uint8'))
            
            self.idx += 1

        del rays_NDC, rays_dir, rays_pts, volume_feature
        self.validation_step_outputs.append(log)   # 新增：手动记录
        return log

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        # 🔥 核心判断：如果没有数据，直接返回，不执行后面的数学运算
        if not outputs:
            return

        # 使用 try-except 或者判断来确保安全
        try:
            mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
            self.log('val/PSNR', mean_psnr, prog_bar=True)

            if self.args.with_depth:
                mask_sum = torch.stack([x['mask_sum'] for x in outputs]).sum()
                # 🔥 再次判断：防止除以 0
                if mask_sum > 0:
                    mean_d_loss_r = torch.stack([x['val_depth_loss_r'] for x in outputs]).mean()
                    mean_abs_err = torch.stack([x['val_abs_err'] for x in outputs]).sum() / mask_sum
                    # ... 其他 acc 计算 ...
                    self.log('val/abs_err', mean_abs_err)
        except Exception as e:
            print(f"Warning: Error during validation epoch end: {e}")

        # 🔥 无论是否计算成功，都要清空，否则内存会爆
        self.validation_step_outputs.clear()


    def save_ckpt(self, name='latest'):
        save_dir = f'runs_new/{self.args.expname}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            'global_step': self.global_step,
            'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            'network_mvs_state_dict': self.MVSNet.state_dict()}
        if self.render_kwargs_train['network_fine'] is not None:
            ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        print('Saved checkpoints at', path)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = MVSSystem(args)
    checkpoint_callback = ModelCheckpoint(os.path.join(f'runs_new/{args.expname}/ckpts/','{epoch:02d}'),
                                          monitor='val/PSNR',
                                          mode='max',
                                          save_top_k=5,
                                          save_last=True)

    logger = TensorBoardLogger(
        save_dir="runs_new",
        name=args.expname
    )

    args.num_gpus, args.use_amp = 1, False
    # 1. 如果你代码里定义了 checkpoint_callback，确保它在 callbacks 列表里
    trainer = Trainer(
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],      # 这里是关键：checkpoint_callback 移到这里
        val_check_interval= args.val_check_interval,
        check_val_every_n_epoch= args.check_val_every_n_epoch,
        accelerator='gpu',                    # 新版 PL 用 accelerator='gpu' 代替 gpus
        devices=1,                            # 指定 1 个 GPU
        num_sanity_val_steps=1,
        benchmark=True,
        precision='bf16-mixed',                        # 如果之前是 16 且显存够，建议先用 32 稳一点
    )

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()
