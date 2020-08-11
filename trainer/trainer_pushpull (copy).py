import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
import time
import os

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import scipy.interpolate

import matplotlib.pyplot as plt


class TrainerPushPull(BaseTrainer):
    """
    Trainer for feature selection using
    convex combination regularization - alternating optimization

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, make_plots=True):
        super().__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.make_plots = make_plots

        # reconstruction loss
        self.train_loss = []
        self.valid_loss = []

        # keep samples for plotting
        self.train_z = None
        self.valid_z = None

        self.block_keep = 1 - self.config['arch']['args']['block_drop']
        assert 0 <= self.block_keep <= 1

        if "plotz" in self.config['trainer']:
            self.plotz = self.config['trainer']['plotz']
        else:
            self.plotz = False

        if "pp_start_epoch" in self.config['loss']:
            self.pp_start_epoch = self.config['loss']['pp_start_epoch']
        else:
            self.pp_start_epoch = 0

        if "reg_coeff" in self.config['loss']:
            self.reg_coeff = self.config['loss']['reg_coeff']
        else:
            self.reg_coeff = 0.0

        if "l1_coeff" in self.config['loss']:
            self.l1_coeff = self.config['loss']['l1_coeff']
        else:
            self.l1_coeff = 0.0

        if "jeffreys_batch" in self.config['loss']:
            self.jeffreys_batch = self.config['loss']['jeffreys_batch']
        else:
            self.jeffreys_batch = 1

        if "jeffreys_vars" in self.config['loss']:
            self.jeffreys_vars = self.config['loss']['jeffreys_vars']
        else:
            self.jeffreys_vars = self.model.num_features

        # learning rate scheduler type
        if self.lr_scheduler is not None:
            self.lr_type = self.config['lr_scheduler']['type']
        else:
            self.lr_type = None


    def _onestep_loss(self, data, target, enable_block_drop=True):
        if self.model.training:
            self.optimizer.zero_grad()

        # get z with no dropout, apply dropout for likelihood term only
        z = self.model._encode(data)
        output = self.model._predict(z, apply_block_drop=enable_block_drop)
        loss = self.loss(output, target)

        # l1 reg
        if self.l1_coeff > 0:
            loss += self.l1_coeff*torch.mean(torch.norm(z, p=1, dim=1))

        # reshape reg
        if self.reg_coeff > 0:
            num_features = self.model.num_features
            batch_size = z.shape[0]

            if num_features == 1:
                # only 1 feature - full info regularization
                perm = np.random.permutation(batch_size)
                z1 = Variable(z.new(size=(batch_size, z.shape[1],)))
                z2 = Variable(z.new(size=(batch_size, z.shape[1],)))

                z1 = z[:,:,0].clone()
                z2 = z[perm,:,0].clone()
                distance = (z1 - z2).norm(p=2, dim=1)
                divergence = self.model.divergence(z1, z2)
                loss += self.reg_coeff*((distance - divergence)**2).mean()

            elif self.jeffreys_batch == 1: 
                # Use 1 sample for expected jeffreys
                z1 = Variable(z.new(size=(batch_size*self.jeffreys_vars, z.shape[1],)))
                z2 = Variable(z.new(size=(batch_size*self.jeffreys_vars, z.shape[1],)))
                distance = Variable(z.new(size=(batch_size*self.jeffreys_vars,)))
                z_1 = Variable(z.new(size=(batch_size*self.jeffreys_vars, z.shape[1], z.shape[2])))
                z_2 = Variable(z.new(size=(batch_size*self.jeffreys_vars, z.shape[1], z.shape[2])))

                for i,feat in enumerate(np.random.choice(num_features, self.jeffreys_vars, replace=False)):
                    # z_i and z_i tilde
                    perm = np.random.permutation(batch_size)

                    z1[i*batch_size:(i+1)*batch_size] = z[:,:,feat].clone()
                    z2[i*batch_size:(i+1)*batch_size] = z[perm,:,feat].clone()
                    distance[i*batch_size:(i+1)*batch_size] = (z1[i*batch_size:(i+1)*batch_size] - z2[i*batch_size:(i+1)*batch_size]).norm(p=2, dim=1)

                    # estimate conditional expectation of D_J                    
                    z_1[i*batch_size:(i+1)*batch_size] = z[:]
                    z_1[i*batch_size:(i+1)*batch_size,:,feat] = z1[i*batch_size:(i+1)*batch_size]

                    z_2[i*batch_size:(i+1)*batch_size] = z[:]
                    z_2[i*batch_size:(i+1)*batch_size,:,feat] = z2[i*batch_size:(i+1)*batch_size]

                # compute regularization term
                loss += self.reg_coeff*((distance-self.model.divergence(z_1, z_2))**2).sum()/batch_size
                '''
                for i in np.random.choice(num_features, self.jeffreys_vars, replace=False):
                    # z_i and z_i tilde
                    perm = np.random.permutation(batch_size)
                    z1 = Variable(z.new(size=(batch_size, z.shape[1],)))
                    z2 = Variable(z.new(size=(batch_size, z.shape[1],)))

                    z1 = z[:,:,i].clone()
                    z2 = z[perm,:,i].clone()
                    distance = (z1 - z2).norm(p=2, dim=1)

                    # estimate conditional expectation of D_J
                    z_1 = Variable(z.new(z.shape))
                    z_2 = Variable(z.new(z.shape))
                    
                    perm_j = np.random.permutation(batch_size)
                    z_1[:] = z[perm_j]
                    z_1[:,:,i] = z1

                    z_2[:] = z[perm_j]
                    z_2[:,:,i] = z2

                    # compute regularization term
                    loss += self.reg_coeff*((distance - self.model.divergence(z_1, z_2))**2).mean()
                '''
            else:
                for i in np.random.choice(num_features, self.jeffreys_vars, replace=False):
                    # z_i and z_i tilde
                    perm = np.random.permutation(batch_size)
                    z1 = Variable(z.new(size=(batch_size, z.shape[1],)))
                    z2 = Variable(z.new(size=(batch_size, z.shape[1],)))

                    z1 = z[:,:,i].clone()
                    z2 = z[perm,:,i].clone()
                    distance = (z1 - z2).norm(p=2, dim=1)

                    # estimate conditional expectation of D_J
                    z_1 = Variable(z.new(size=(batch_size*self.jeffreys_batch, z.shape[1], z.shape[2])))
                    z_2 = Variable(z.new(size=(batch_size*self.jeffreys_batch, z.shape[1], z.shape[2])))
                    for j in range(self.jeffreys_batch):
                        perm_j = np.random.permutation(batch_size)
                        z_1[j*batch_size:(j+1)*batch_size] = z[:]
                        z_1[j*batch_size:(j+1)*batch_size,:,i] = z1

                        z_2[j*batch_size:(j+1)*batch_size] = z[:]
                        z_2[j*batch_size:(j+1)*batch_size,:,i] = z2

                    jeffreys = self.model.divergence(z_1, z_2)
                    
                    jeffrey_est = Variable(jeffreys.new(size=(batch_size,)))
                    for j in range(batch_size):
                        jeffrey_est[j] = jeffreys[j::batch_size].mean()

                    # compute regularization term
                    loss += self.reg_coeff*((distance - jeffrey_est)**2).mean()

        loss += 2.0
        if self.model.training:
            loss.backward()
            self.optimizer.step()

        return loss, output, z


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # loss,output,z = self._recon_loss(data, target, n_runs=self.predict_runs)
            loss, output, z = self._onestep_loss(data, target, enable_block_drop=True)

            # evaluate metrics; logging
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

            # sample z values for plotting
            if batch_idx == 0 and self.plotz:
                self.train_z = z.detach().cpu().numpy()
        
            total_loss += loss.item()

        # log losses separately
        self.train_loss.append(total_loss/len(self.data_loader))

        # log sum of all losses
        log = {
            'loss': total_loss/len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        # for use with e.g. StepLR
        if self.lr_type == 'StepLR':
            self.lr_scheduler.step()

        # make plots, if applicable
        if self.make_plots:
            self._make_plots(save=epoch%10)

        return log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # loss, output, z = self._recon_loss(data, target)
                loss, output, z = self._onestep_loss(data, target, enable_block_drop=False)

                # evaluate metrics; logging
                total_val_metrics += self._eval_metrics(output, target)

                if batch_idx == 0 and self.plotz:
                    self.valid_z = z.detach().cpu().numpy()

                total_val_loss += loss.item()

        # log losses separately
        self.valid_loss.append(total_val_loss/len(self.valid_data_loader))

        # for use with e.g. ReduceLROnPlateau
        if self.lr_type == 'ReduceLROnPlateau':
            self.lr_scheduler.step(total_val_loss / len(self.valid_data_loader))

        return {
            'val_loss': total_val_loss/len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

 
    def _eval_metrics(self, output, target):
        return_list = []
        for metric in self.metrics:
            return_list.append(metric(output, target))
        return return_list


    def _make_plots(self, save=False):
        '''
            Run %matplotlib notebook
        '''
        if not hasattr(self, 'fig'):
            if self.plotz:
                self.fig,(self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(8,3))
            else:
                self.fig, self.ax1 = plt.subplots(figsize=(4,3))
            
            plt.ion()
            self.fig.show()
            self.fig.canvas.draw()

        self.ax1.clear()

        # reconstruction loss
        epochs = np.arange(len(self.train_loss))+1
        self.ax1.semilogy(epochs, self.train_loss, '--', label="Training Loss")
        if len(self.valid_loss) > 0:
            self.ax1.semilogy(epochs, self.valid_loss, '--', label="Validation Loss")

        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax1.legend()
        self.ax1.grid(True, which="both")

        if self.plotz:
            self.ax2.clear()
            idx = np.random.choice(self.train_z.shape[2])
            if self.train_z.shape[1] > 1:
                self.ax2.scatter(self.train_z[:,0], self.train_z[:,1], label="Training Z")
            else:
                self.ax2.hist(self.train_z[:,0,idx], density=True, label="Training Z: dim: %d"%idx)

            if len(self.valid_loss) > 0:
                if self.valid_z.shape[1] > 1:
                    self.ax2.scatter(self.valid_z[:,0], self.valid_z[:,1], label="Validation Z")
                else:
                    self.ax2.hist(self.valid_z[:,0,idx], density=True, label="Validation Z: dim: %d"%idx)

            self.ax2.legend()

        # draw and sleep
        plt.tight_layout()
        self.fig.canvas.draw()
        if save:
            plt.savefig(os.path.join(self.checkpoint_dir, "train_valid_plot.png"))
