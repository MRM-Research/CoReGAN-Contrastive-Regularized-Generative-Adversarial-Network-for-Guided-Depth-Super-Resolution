import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
from .misc import un_tan_fi
from collections import OrderedDict
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from .lr_scheduler import MultiStepRestartLR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .loss import compute_gradient_penalty
from .loss import GANLoss, MSELoss
from .loss import custom_loss

loss_dict = OrderedDict()

def get_current_visuals(self):
    out_dict = OrderedDict()
    out_dict['depth_low_res'] = self.depth_low_res.detach().to(self.device)
    out_dict['result'] = self.output.detach().to(self.device)
    out_dict['RGB'] = self.RGB.detach().to(self.device)
    if hasattr(self, 'depth_high_res'):
        out_dict['depth_high_res'] = self.depth_high_res.detach().to(self.device)
    return out_dict

P = PeakSignalNoiseRatio()
Z = StructuralSimilarityIndexMeasure()

def calculate_metrics(self, img1, img2):
    # revert both images to 0, 1 from -1, 1
    img1 = un_tan_fi(img1)
    img2 = un_tan_fi(img2)
    
    MSE_metric = mean_squared_error(img1, img2)
    MAE_metric = mean_absolute_error(img1, img2)
    
    # for mse, mae check range of calc
    # for psnr, x by 255
    
    img1 = img1*255
    img1 = img1.round().int()
    img1 = img1.float()

    img2 = img2*255
    img2 = img2.round().int()
    img2 = img2.float()

    return P(img1, img2).to(self.device), Z(img1, img2).to(self.device), MSE_metric.to(self.device), MAE_metric.to(self.device)

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Epoch:
    def __init__(self, gan_type, model, loss, stage_name, device="cpu", verbose=True):
        self.net_g = model
        self.loss = loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.gan_type = gan_type
        
        self.GLoss = GANLoss(gan_type=self.gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1)
        self.MLoss = MSELoss(loss_weight=1, reduction='mean')
        self._to_device()

    def _to_device(self):
        self.net_g.to(self.device)
        self.GLoss.to(self.device)
        self.MLoss.to(self.device)


    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self,iter,x,z,y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {"MSE": AverageValueMeter(), "MAE": AverageValueMeter(), "PSNR": AverageValueMeter(), "SSIM": AverageValueMeter()}
        iter = 0
        l_g_total = 0
        loss_dict = OrderedDict()
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x,z,y in iterator:    
                x, z, y = x.to(self.device), z.to(self.device), y.to(self.device)
                loss, ssim, psnr, mae, mse = self.batch_update(iter,x,z,y) ### log both? how?

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                metrics_meters["MSE"].add(mse.cpu().detach().numpy())
                metrics_meters["MAE"].add(mae.cpu().detach().numpy())
                metrics_meters["PSNR"].add(psnr.cpu().detach().numpy())
                metrics_meters["SSIM"].add(ssim.cpu().detach().numpy())

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
                iter = iter + 1

        return logs

class TrainEpoch(Epoch):
    def __init__(self, model, loss, discriminator, loss_weight_loss, loss_weight_gan, device="cpu", verbose=True, contrastive=True,gan_type = "standard"):
        super().__init__(
            model=model,
            loss=loss,
            gan_type=gan_type,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.loss_weight_loss = loss_weight_loss
        self.loss_weight_gan = loss_weight_gan
        self.net_g = model
        self.net_d = discriminator
        self.contrastive = contrastive
        self.schedulers = []

        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),
                                    lr=0.0001, weight_decay=0, betas=[0.9, 0.99])

        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                    lr=0.0001, weight_decay=0, betas=[0.9, 0.99])

    def on_epoch_start(self):
        self.net_g.train()
        self.net_d.train()
        
    def feed_data(self,x,z,y):
        self.rgb = x
        self.depth_high_res = y
        self.depth_low_res = z
    
    def batch_update(self, current_iter,x,z,y):
        
        self.feed_data(x,z,y)
       
        # creating a list of optimizers to allow integration of lr_scheduler
        self.optimizers = [self.optimizer_g, self.optimizer_d]
        
        self.net_d_iters = 1
        self.net_d_init_iters = 0

        for optimizer in self.optimizers:
            self.schedulers.append(
                MultiStepRestartLR(optimizer, milestones=[50000, 100000, 200000, 300000], gamma=0.5))

        for p in self.net_d.parameters():
            p.requires_grad = False
        
        # setting net_g gradients to zero
        self.optimizer_g.zero_grad()

        # generating output
        self.output,f1,f2 = self.net_g(self.rgb, self.depth_low_res)
        print(type(self.output))
        print(type(self.depth_high_res))
        
        # initiliazing MSELoss from Epoch
        #self.cri_pix = self.MLoss(self.output, self.depth_high_res).to(self.device)
        #print("cri_pix - ", self.cri_pix)
        
        # initializing GANLoss from Epoch
        #self.cri_gan = self.GLoss(fake_g_pred, True, is_disc=False).to(self.device)

        l_g_total = 0
        loss_dict = OrderedDict()           

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            # pixel loss
            l_g_pix = self.MLoss(self.output, self.depth_high_res).to(self.device)
            print("l_g_pix - ", l_g_pix)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.GLoss(fake_g_pred, True, is_disc=False).to(self.device)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # backprop for generator
            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        # setting net_d gradients to zero
        self.optimizer_d.zero_grad()

        # generating output
        self.output = self.net_g(self.rgb, self.depth_low_res)
        
        # real image generation
        real_d_pred = self.net_d(self.depth_high_res)
        l_d_real = self.GLoss(real_d_pred, True, is_disc=True).to(self.device)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        # fake image generation
        fake_d_pred = self.net_d(self.output)
        l_d_fake = self.GLoss(fake_g_pred, True, is_disc=True).to(self.device)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # gradient penalty for discriminator
        self.gp_weight = 100
        gradient_penalty = compute_gradient_penalty(self.net_d, self.depth_high_res, self.output, self.device)
        l_d = l_d_real + l_d_fake + (self.gp_weight * gradient_penalty)

        # backprop for discriminator
        l_d.backward()
        self.optimizer_d.step()       
        
        visuals = self.get_current_visuals()
        guiding_img = visuals['RGB'] 
        result_img = visuals['result']
        input_img = visuals['depth_low_res']
        if 'depth_high_res' in visuals:
            DHR_img = visuals['depth_high_res']
            del self.y
      
        psnr, ssim, mse_metric, mae_metric = self.calculate_metrics(result_img, DHR_img)

        loss = custom_loss(self.output,DHR_img, f1, f2)   

        return loss, psnr, ssim, mse_metric, mae_metric

class ValidEpoch(Epoch):
    def __init__(self, model, optimizer, device="cpu", verbose=True, contrastive=False):
        super().__init__(
            model=model,
            stage_name="valid",
            device=device,
            verbose=verbose,           
        )
        self.contrastive = contrastive
        self.optimizer = optimizer,

    def on_epoch_start(self):
        self.net_g.eval()

    # def batch_update(self, x,z, y):
    #     with torch.no_grad():
    #         if self.contrastive:
    #             prediction, ft1, ft2 = self.model.forward(x,z)
    #             loss = self.loss(prediction, y, ft1, ft2)
    #         else:
    #             prediction = self.model.forward(x,z)
    #             loss = self.loss(prediction, y)
    #     return loss, prediction

    def batch_update(self, current_iter):
       
        # creating a list of optimizers to allow integration of lr_scheduler
        self.optimizers = [self.optimizer_g, self.optimizer_d]

        # initiliazing MSELoss from Epoch
        cri_pix_cls = self.MLoss
        self.cri_pix = cri_pix_cls(loss_weight=self.loss_weight, reduction='mean').to(self.device)
        
        # initializing GANLoss from Epoch
        cri_gan_cls = self.GLoss
        self.cri_gan = cri_gan_cls(gan_type='standard', real_label_val=1.0, fake_label_val=0.0, loss_weight=1).to(self.device)
        self.gp_weight = 100

        self.net_d_iters = 1
        self.net_d_init_iters = 0

        for optimizer in self.optimizers:
            self.schedulers.append(
                MultiStepRestartLR(optimizer, milestones=[50000, 100000, 200000, 300000], gamma=0.5))

        for p in self.net_d.parameters():
            p.requires_grad = False
        
        # setting net_g gradients to zero
        self.optimizer_g.zero_grad()

        # generating output
        self.output = self.net_g(self.RGB, self.depth_low_res)

        l_g_total = 0
        loss_dict = OrderedDict()           

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.depth_high_res)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        # setting net_d gradients to zero
        self.optimizer_d.zero_grad()

        # generating output
        self.output = self.net_g(self.RGB, self.depth_low_res)
        
        # real image generation
        real_d_pred = self.net_d(self.depth_high_res)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        # fake image generation
        fake_d_pred = self.net_d(self.output)
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # gradient penalty for discriminator
        gradient_penalty = compute_gradient_penalty(self.net_d, self.depth_high_res, self.output, self.device)
        l_d = l_d_real + l_d_fake + (self.gp_weight * gradient_penalty)     
        
        visuals = self.get_current_visuals()
        input_img = visuals['RGB'] 
        result_img = visuals['result']
        if 'depth_high_res' in visuals:
            DHR_img = visuals['depth_high_res']
            del self.depth_high_res
      
        psnr, ssim, mse_metric, mae_metric = self.calculate_metrics(result_img, DHR_img)   

        return l_g_total, psnr, ssim, mse_metric, mae_metric
