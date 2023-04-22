import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
import utils
import scipy.sparse as sp
import numpy as np


class DiscreteDenoisingDiffusion(pl.LightningModule):
    """

    Discrete Denoising Diffusion (DiGress from https://github.com/cvignac/DiGress, jcognac)
    Edge Predictive Discrete Denoising Diffusion (edge-DiGress from HiGGs, anonymous authors)

    """
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)


        try:
            # During sampling we can't load the hyperparameters, but don't need them, so just pass the exception
            self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
            print(f"finished saving hyperparameters")
        except:
            print("="*100+f"\nFailed to save hyperparams, assuming this is for sampling\n" +"="*100)
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0



    def training_step(self, data, i):

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        # E is shape (batch size x nodes x nodes x classes)
        X, E = dense_data.X, dense_data.E

        if self.cfg.dataset.h != 1.5:
            # Not so complicated when just doing h1 or h2
            noisy_data = self.apply_noise(X, E, data.y, node_mask)
        else:
            # Different noise application process for edge prediction
            # Additionally, we need to preserve the class of graph and the node categories
            noisy_data = self.apply_noise_glue(X, E, data.y, node_mask, data.pos)
            noisy_data['X_t'] = dense_data.X
            noisy_data['y_T'] = data.y

        # Compute extra features (cycles etc) and predict clean graph
        extra_data = self.compute_extra_data(noisy_data, data.pos)
        pred = self.forward(noisy_data, extra_data, node_mask)

        if self.cfg.dataset.h != 1.5:
            # Loss etc is easy for h1, h2
            loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                                   true_X=X, true_E=E, true_y=data.y,
                                   log=i % self.log_every_steps == 0)
            self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                               log=i % self.log_every_steps == 0)
        else:
            # Need to re-compute h1-match matrix and reset edges and node categories where appropriate
            bs = E.size(0)
            N = E.size(1)
            pos_dim = data.pos.shape
            pos = data.pos.view(bs, pos_dim[1], pos_dim[1])
            pos = pos[:, :N, :N]
            pos_dim = pos.shape
            pos = pos.expand(E.size(-1), bs, pos_dim[1], pos_dim[1])
            pos = torch.permute(pos, (1, 2, 3, 0))

            # Only want to compute loss on relevant edges (doesn't need to learn to de-noise within h1 graphs)
            pred.E[pos] = -1.
            pred.X[~node_mask] = -1.
            loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=data.y,
                                   true_X=X, true_E=E, true_y=data.y,
                                   log=i % self.log_every_steps == 0)
            self.train_metrics(masked_pred_X=X, masked_pred_E=pred.E, true_X=X, true_E=E,
                               log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        self.train_metrics.log_epoch_metrics(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_y_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        if self.cfg.dataset.h != 1.5:
            # Again noise is easy for h1, h2
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        else:
            # Specific noise for edge-DiGress, keep node and graph categories
            noisy_data = self.apply_noise_glue(dense_data.X, dense_data.E, data.y, node_mask, data.pos)
            noisy_data['X_t'] = dense_data.X
            noisy_data['y_t'] = data.y
        extra_data = self.compute_extra_data(noisy_data, data.pos)
        pred = self.forward(noisy_data, extra_data, node_mask)

        if self.cfg.dataset.h == 1.5:
            # Need to reset node and graph categories, as well as intra-h1 edges
            bs = pred.E.size(0)
            N = pred.E.size(1)
            pos_dim = data.pos.shape
            pos = data.pos.view(bs, pos_dim[1], pos_dim[1])
            pos = pos[:, :N, :N]
            pos_dim = pos.shape
            pos = pos.expand(pred.E.size(-1), bs, pos_dim[1], pos_dim[1])
            pos = torch.permute(pos, (1, 2, 3, 0))
            pred.E[pos] = dense_data.E[pos]

            pred.y = data.y

        if self.cfg.dataset.h != 1.5:
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False)
            return {'loss': nll, 'data':None}
        else:
            # Need to pass dense data forward for edge prediction
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False, pos=data.pos)
            return {'loss': nll, 'data': data, 'bs':dense_data.X.shape[0]}

    def validation_epoch_end(self, outs) -> None:
        """
        Compute sampling metrics on a set of sampled graphs.
        Modified to include edge-DiGress.
        """
        metrics = [self.val_nll.compute(), self.val_X_kl.compute(), self.val_E_kl.compute(),
                   self.val_y_kl.compute(), self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        wandb.log({"val/epoch_NLL": metrics[0],
                   "val/X_kl": metrics[1],
                   "val/E_kl": metrics[2],
                   "val/y_kl": metrics[3],
                   "val/X_logp": metrics[4],
                   "val/E_logp": metrics[5],
                   "val/y_logp": metrics[6]}, commit=False)

        print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
              f"Val Edge type KL: {metrics[2] :.2f} -- Val Global feat. KL {metrics[3] :.2f}\n")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        # Sample graphs (from input where appropriate)
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save
            samples = []
            ident = 0
            while samples_left_to_generate > 0:

                if self.cfg.dataset.h != 1.5:
                    # More simple for h1 and h2 as we don't need to pass data forwards for sampling
                    bs = 2 * self.cfg.train.batch_size
                    to_generate = min(samples_left_to_generate, bs)
                    to_save = min(samples_left_to_save, bs)
                    chains_save = min(chains_left_to_save, bs)
                    samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                     save_final=to_save,
                                                     keep_chain=chains_save,
                                                     number_chain_steps=self.number_chain_steps))
                else:
                    # Pass pairs of h1 graphs forwards from the validation epoch
                    # N.B. limited to the validation batch size
                    bs = outs[0]['bs']
                    to_generate = bs
                    to_save = min(samples_left_to_save, bs)
                    chains_save = min(chains_left_to_save, bs)
                    samples.extend(self.sample_batch_glue(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                     save_final=to_save,
                                                     keep_chain=chains_save,
                                                     number_chain_steps=self.number_chain_steps,
                                                     data = outs[0]['data']))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            print("Computing sampling metrics...")
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_y_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()

    def test_step(self, data, i):
        # Very similar to validation epochs - see above
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        if self.cfg.dataset.h != 1.5:
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        else:
            noisy_data = self.apply_noise_glue(dense_data.X, dense_data.E, data.y, node_mask, data.pos)
            noisy_data['X_t'] = dense_data.X
            noisy_data['y_t'] = data.y
        extra_data = self.compute_extra_data(noisy_data, data.pos)
        pred = self.forward(noisy_data, extra_data, node_mask)

        if self.cfg.dataset.h == 1.5:
            bs = pred.E.size(0)
            N = pred.E.size(1)
            pos_dim = data.pos.shape
            pos = data.pos.view(bs, pos_dim[1], pos_dim[1])
            pos = pos[:, :N, :N]
            pos_dim = pos.shape
            pos = pos.expand(pred.E.size(-1), bs, pos_dim[1], pos_dim[1])
            pos = torch.permute(pos, (1, 2, 3, 0))

            pred.E[pos] = -1.
            pred.X[~node_mask] = -1.
            pred.y = data.y

        if self.cfg.dataset.h != 1.5:
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False)
            return {'loss': nll, 'data':None}
        else:
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False, pos=data.pos)
            return {'loss': nll, 'data': data, 'bs':dense_data.X.shape[0]}

    def test_epoch_end(self, outs) -> None:
        """
        Measure likelihood on a test set and compute stability metrics.
        Again modified to include edge-DiGress, see validation_epoch_end
        """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_y_kl.compute(), self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]
        wandb.log({"test/epoch_NLL": metrics[0],
                   "test/X_mse": metrics[1],
                   "test/E_mse": metrics[2],
                   "test/y_mse": metrics[3],
                   "test/X_logp": metrics[4],
                   "test/E_logp": metrics[5],
                   "test/y_logp": metrics[6]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f} -- Test Global feat. KL {metrics[3] :.2f}\n")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.samples_to_generate
        samples_left_to_save = self.cfg.general.samples_to_save
        chains_left_to_save = self.cfg.general.chains_to_save

        samples = []

        ident = 0
        while samples_left_to_generate > 0:

            if self.cfg.dataset.h != 1.5:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
            else:
                bs = outs[0]['bs']  # 2 * self.cfg.train.batch_size
                to_generate = bs  # min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch_glue(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                      save_final=to_save,
                                                      keep_chain=chains_save,
                                                      number_chain_steps=self.number_chain_steps,
                                                      data=outs[0]['data']))
            ident += to_generate

            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()
        # print("Done.")


    def kl_prior(self, X, E, y, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        proby = y @ Qtb.y if y.numel() > 0 else y
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        uniform_dist_y = torch.ones_like(proby) / self.ydim_output

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        kl_distance_y = F.kl_div(input=proby.log(), target=uniform_dist_y, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E) + \
               diffusion_utils.sum_except_batch(kl_distance_y)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        kl_y = (self.test_y_kl if test else self.val_y_kl)(prob_true.y, torch.log(prob_pred.y)) if pred_probs_y.numel() != 0 else 0
        return kl_x + kl_e + kl_y

    def reconstruction_logp(self, t, X, E, y, node_mask, pos = None):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)
        probY0 = y @ Q0.y if y.numel() > 0 else y

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}

        if pos is None:
            extra_data = self.compute_extra_data(noisy_data)
        else:
            extra_data = self.compute_extra_data(noisy_data, pos = pos)

        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """
        Sample noise and apply it to the data.
        This is the vanilla (non-edge-prediction) version.
        """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def apply_noise_glue(self, X, E, y, node_mask, pos):
        """
        Sample noise and apply it to the data.
        This is modified heavily for edge-DiGress.
        """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        # We actually want to fix X (node attributes)
        probX = X
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        # Get "are they from the same h1 graph" matrix
        bs = E.size(0)
        N = E.size(1)
        pos_dim = pos.shape
        pos = pos.view(bs, pos_dim[1], pos_dim[1])
        pos = pos[:,:N,:N]
        pos_dim = pos.shape
        pos = pos.expand(E.size(-1), bs, pos_dim[1], pos_dim[1])
        pos = torch.permute(pos, (1,2,3,0))

        # Reset edges that aren't inter-h1
        probE[pos] = E[pos]

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False, pos = None):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, y, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        if pos is None:
            prob0 = self.reconstruction_logp(t, X, E, y, node_mask)
        else:
            prob0 = self.reconstruction_logp(t, X, E, y, node_mask, pos=pos)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log()) + \
                      self.val_y_logp(y * prob0.y.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        wandb.log({"kl prior": kl_prior.mean(),
                   "Estimator loss terms": loss_all_t.mean(),
                   "log_pn": log_pN.mean(),
                   "loss_term_0": loss_term_0,
                   'test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, data=None):
        """
        Sample a batch of graphs.
        Does not cover edge-DiGress (see sample_batch_glue)

        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: graph_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int==100)
            X, E, y = sampled_s.X, sampled_s.E, y#sampled_s.y
            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            graph_list.append([atom_types, edge_types])
            if i < 3:
                print("Example of generated E: ", edge_types)
                print("Example of generated X: ", atom_types)

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            predicted_graph_list.append([atom_types, edge_types])


        # Visualize chains
        if self.visualization_tools is not None:
            # print('Visualizing chains...')
            current_path = os.getcwd()
            num_graphs = chain_X.size(1)       # number of graphs
            for i in range(num_graphs):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/graph_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())

            # Visualize the final graphs
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, predicted_graph_list, min(len(predicted_graph_list), 15), log='predicted')
            try:
                self.visualization_tools.visualize_grid(result_path, predicted_graph_list, min(len(predicted_graph_list), 15), log='predicted_grid')
            except:
                pass
        return graph_list

    @torch.no_grad()
    def sample_batch_glue(self, batch_id: int, batch_size: int,
                          keep_chain: int, number_chain_steps: int, data=None):
        """
        Sample a batch of edge-predictions between two h1 graphs (with those graphs contained in data)

        :param batch_id: int
        :param batch_size: int
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain

        :param data: batch of graph pairs and their connections

        :return: graph_list. Each element of this list is a tuple (node_types, charges, positions)
        """
        # # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        n_nodes = data.num_nodes
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

        X = dense_data.X
        E = dense_data.E
        y = z_T.y

        pos = data.pos
        bs = E.size(0)
        N = E.size(1)
        pos_dim = pos.shape
        pos = pos.view(bs, pos_dim[1], pos_dim[1])
        pos = pos[:,:N,:N]
        pos_dim = pos.shape
        pos = pos.expand(E.size(-1), bs, pos_dim[1], pos_dim[1])
        pos = torch.permute(pos, (1,2,3,0))

        #=========================================
        E[~pos] = 0.
        #=========================================

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int==100, pos = data.pos)
            #========================================================
            E[~pos] = sampled_s.E[~pos]
            #========================================================

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            discrete_sampled_s.X = torch.argmax(X, dim=-1)
            discrete_sampled_s.E = torch.argmax(E, dim = -1)
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)

        pos = pos.clone()[:,:,:,0]
        E_save = torch.clone(E)
        E_save = torch.argmax(E_save, dim=-1)

        #========================================================================
        E = sampled_s.E
        E[pos] = E_save[pos]
        #===========================================================================

        X = dense_data.X
        X = torch.argmax(X, dim=-1)

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        graph_list = []
        for i in range(batch_size):
            n = n_nodes
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            graph_list.append([atom_types, edge_types])
            if i < 3:
                print("Example of generated E: ", edge_types)
                print("Example of generated X: ", atom_types)

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes #[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            predicted_graph_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            # print('Visualizing chains...')
            current_path = os.getcwd()
            num_graphs = chain_X.size(1)       # number of graphs
            for i in range(num_graphs):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/graph_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())

            # Visualize the final graphs
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, predicted_graph_list, min(len(predicted_graph_list), 15), log='predicted')
            try:
                self.visualization_tools.visualize_grid(result_path, predicted_graph_list, min(len(predicted_graph_list), 15), log='predicted_grid')
            except:
                pass
        return graph_list

    def sampling_batch_h1(self, batch_id: int, batch_size: int,
                        keep_chain: int, number_chain_steps: int,
                        num_nodes=None, y = None, figures = False):
        """
        Sample a batch of h1 graphs
        If y is passed, uses it to condition that generation


        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain

        :param y: torch.Tensor integer node category
        :param figures: Whether to visualise the sampled graphs

        :return: graph_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

        X, E = z_T.X, z_T.E

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int==100)
            X, E, y = sampled_s.X, sampled_s.E, y
            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            graph_list.append([atom_types, edge_types])

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()

            num_comps, comps = sp.csgraph.connected_components(edge_types)
            _, count = np.unique(comps, return_counts=True)
            subset = np.in1d(comps, count.argsort()[-1:])
            predicted_graph_list.append([atom_types[subset], edge_types[subset,:][:, subset]])

        networkx_graphs = []
        if figures:

            for i in range(len(predicted_graph_list)):
                graph = utils.to_networkx(predicted_graph_list[i][0].numpy(), predicted_graph_list[i][1].numpy())
                networkx_graphs.append(graph)

            # Visualize chains
            if self.visualization_tools is not None:


                y_labels = torch.argmax(y, axis = -1).cpu().tolist()
                # print('Visualizing chains...')
                current_path = os.getcwd()
                num_graphs = chain_X.size(1)       # number of graphs
                for i in range(num_graphs):
                    result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                             f'epoch{self.current_epoch}/'
                                                             f'chains/graph_{batch_id + i}')
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        _ = self.visualization_tools.visualize_chain(result_path,
                                                                     chain_X[:, i, :].numpy(),
                                                                     chain_E[:, i, :].numpy())
                # Visualize the final graphs
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
                try:
                    self.visualization_tools.visualize_grid(result_path, predicted_graph_list,
                                                        min(len(predicted_graph_list), 15), log='Sampled Grid H1',
                                                        labels = y_labels, largest_component=False)
                except:
                    pass

        return predicted_graph_list, networkx_graphs

    def sampling_batch_glue(self, batch_id: int, batch_size: int, keep_chain: int,
                            number_chain_steps: int, data=None, figures = False):
        """
        Sample a batch of edge predictions between h1 graphs

        :param batch_id: int
        :param batch_size: int
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain

        :param data: pairs of h1 graphs to predict edges between
        :param figures: Whether to visualise the sampled graphs

        :return: graph_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        # # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        n_nodes = data.num_nodes
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X = dense_data.X
        E = dense_data.E
        y = data.y

        pos = data.pos
        bs = E.size(0)
        N = E.size(1)
        pos_dim = pos.shape
        pos = pos.view(bs, pos_dim[1], pos_dim[1])

        if pos_dim[1] > N:
            pos = pos[:,:N,:N]
        pos_dim = pos.shape
        pos = pos.expand(E.size(-1), bs, pos_dim[1], pos_dim[1])
        pos = torch.permute(pos, (1,2,3,0))

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            sampled_s, discrete_sampled_s, predicted_graph, prob_E = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int==100, pos = data.pos,
                                                                                       return_probE = True)
            E[~pos] = sampled_s.E[~pos]
            write_index = (s_int * number_chain_steps) // self.T

            discrete_sampled_s.X = torch.argmax(X, dim=-1)
            discrete_sampled_s.E = torch.argmax(E, dim = -1)

            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        # recover E just in case no edges are predicted
        # recover_E = torch.clone(sampled_s).mask(node_mask, collapse = False)
        sampled_s = sampled_s.mask(node_mask, collapse=True)

        pos_save = pos.clone()
        pos = pos.clone()[:,:,:,0]
        E_save = torch.clone(E)
        # We know in sampling that there should be at least one edge
        E_save = torch.argmax(E_save, dim=-1)

        # =================================
        prob_E[pos_save] = 0.
        # =================================

        # E, sampled_s.E are already argmaxed during masking
        E = sampled_s.E
        E[pos] = E_save[pos]

        X = torch.argmax(dense_data.X, dim=-1)
        X = X >= 0
        max_size = torch.sum(X)

        # ======================================================================

        adj_inter_community = torch.clone(E)
        adj_inter_community[pos] = 0.
        for ib in range(batch_size):

            # Check that there is at least one edge between the graphs
            if 1. not in adj_inter_community[ib, :, :] and 1 not in adj_inter_community[ib, :, :]:
                # If there are no edges, take the most likely one
                options = prob_E[ib, :, :, 1:]
                options[n_nodes:, n_nodes:] = 0.
                options[max_size:, max_size:, :] = 0.
                options[options == 0.5] = 0.
                max_value = torch.max(options)
                max_mask  = options == max_value
                max_inds = torch.argwhere(max_mask)
                ix, iy, iz = max_inds[0, 0], max_inds[0, 1], max_inds[0, 2]

                # +1 to offset slicing earlier
                E[ib, ix, iy] = iz + 1
                E[ib, iy, ix] = iz + 1

        X = dense_data.X
        X = torch.argmax(X, dim=-1)
        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            predicted_graph_list.append([atom_types, edge_types])

        networkx_graphs = []

        for i in range(len(predicted_graph_list)):
            graph = utils.to_networkx(predicted_graph_list[i][0].numpy(), predicted_graph_list[i][1].numpy())
            networkx_graphs.append(graph)

        if figures:
            # Visualize chains
            if self.visualization_tools is not None:
                # print('Visualizing chains...')
                current_path = os.getcwd()
                num_graphs = chain_X.size(1)       # number of graphs
                for i in range(num_graphs):
                    result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                             f'epoch{self.current_epoch}/'
                                                             f'chains/graph_{batch_id + i}')
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        _ = self.visualization_tools.visualize_chain(result_path,
                                                                     chain_X[:, i, :].numpy(),
                                                                     chain_E[:, i, :].numpy())

                # Visualize the final graphs
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
                self.visualization_tools.visualize_grid(result_path, predicted_graph_list, min(len(predicted_graph_list), 15),
                                                        log='Sampled Glue Grid', largest_component=True)

        return predicted_graph_list, networkx_graphs

    def sampling_batch_h2(self, batch_id: int, batch_size: int, keep_chain: int,
                          number_chain_steps: int, num_nodes=None, figures = False):
        """
        Sample a batch of h2 graphs (generally only one is sampled at once, bs=1)

        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain

        :param figures: Whether to visualise the sampled graphs

        :return: graph_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y
        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int==100)
            X, E, y = sampled_s.X, sampled_s.E, y
            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()

            num_comps, comps = sp.csgraph.connected_components(edge_types)
            _, count = np.unique(comps, return_counts=True)
            subset = np.in1d(comps, count.argsort()[-1:])
            predicted_graph_list.append([atom_types[subset], edge_types[subset,:][:, subset]])

        networkx_graphs = []
        for i in range(len(predicted_graph_list)):
            graph = utils.to_networkx(predicted_graph_list[i][0].numpy(), predicted_graph_list[i][1].numpy())
            networkx_graphs.append(graph)

        if figures:
            # Visualize chains
            if self.visualization_tools is not None:
                current_path = os.getcwd()
                num_graphs = chain_X.size(1)       # number of graphs
                for i in range(num_graphs):
                    result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                             f'epoch{self.current_epoch}/'
                                                             f'chains/graph_{batch_id + i}')
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                        _ = self.visualization_tools.visualize_chain(result_path,
                                                                     chain_X[:, i, :].numpy(),
                                                                     chain_E[:, i, :].numpy())

                # Visualize the final graphs
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
                self.visualization_tools.visualize(result_path, predicted_graph_list,
                                                   min(len(predicted_graph_list), 15), log='Sampled H2')

        return predicted_graph_list, networkx_graphs



    def sample_p_zs_given_zt(self, t, X_t, E_t, y_t, node_mask, last_step: bool, pos = None, return_probE = False):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        if pos is None:
            extra_data = self.compute_extra_data(noisy_data)
        else:
            extra_data = self.compute_extra_data(noisy_data, pos = pos)

        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        pred_Y = F.softmax(pred.y, dim = -1)

        if last_step:
            predicted_graph = diffusion_utils.sample_discrete_features(pred_X, pred_E, node_mask=node_mask)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        if pred.y.numel() != 0:
            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, probY=pred_Y, node_mask=node_mask)
        else:
            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        Y_s = F.one_hot(sampled_s.y, num_classes=self.ydim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=Y_s)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=Y_s)

        if return_probE:
            return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), \
                   predicted_graph if last_step else None, prob_E
        else:
            return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), \
                   predicted_graph if last_step else None

    def compute_extra_data(self, noisy_data, pos = None):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        if pos is None:
            extra_features = self.extra_features(noisy_data)
        else:
            extra_features = self.extra_features(noisy_data, pos = pos)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
