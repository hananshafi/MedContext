#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnformer.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnformer.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnformer.utilities.to_torch import maybe_to_torch, to_cuda
from nnformer.network_architecture_nnunet.generic_UNet import Generic_UNet
from nnformer.network_architecture_nnunet.initialization import InitWeights_He
from nnformer.network_architecture_nnunet.neural_network import SegmentationNetwork
from nnformer.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnformer.training.dataloading.dataset_loading import unpack_dataset
from nnformer.training.network_training.nnFormerTrainer_synapse_2 import nnFormerTrainer_synapse
from nnformer.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnformer.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from torch.nn import functional as F


def dist_loss(t, s):
    T = 1.0
    prob_t = F.softmax(t/T, dim=1)
    log_prob_s = F.log_softmax(s/T, dim=1)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss

def entropy_loss(t,s,num_classes=14):
    T = 1.0
    dist_loss = 0
    for i in range(num_classes):
        s_in = s[:,i,:,:,:]
        t_in = t[:,i,:,:,:]
        prob_t = F.softmax(t_in.reshape(t.shape[0],-1)/T, dim=-1)
        log_prob_s = F.log_softmax(s_in.reshape(s.shape[0],-1)/T, dim=-1)
        dist_loss += -(prob_t*log_prob_s).sum(dim=-1).mean()
    return dist_loss / num_classes


def norm_l2_loss(t,s):
    T=0.5
    num_classes = t.shape[1]
    #t = F.softmax(t/T, dim=1).sum(1)
    #s = F.softmax(s, dim=1).sum(1)
    t = F.softmax(t/T, dim=1).permute(0,2,3,4,1).reshape(-1,num_classes)
    s = F.softmax(s, dim=1).permute(0,2,3,4,1).reshape(-1,num_classes)
    loss = 1 - F.cosine_similarity(t,s).mean()
    #loss = 1 - F.cosine_similarity(t.reshape(t.shape[0],-1),s.reshape(s.shape[0],-1)).mean()
    return loss

def KLDivergenceLoss(s,t, num_classes=14):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # Pixelwise KL divergence loss
    loss = 0
    for i in range(num_classes):
        s_in = s[:,i,:,:,:]
        t_in = t[:,i,:,:,:]
        s_in = F.log_softmax (s_in.reshape(s.shape[0],-1), dim=-1)
        t_in = F.softmax (t_in.reshape(t.shape[0],-1), dim=-1)
        loss+=kl_loss(s_in, t_in)
        
    return (loss / num_classes)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


    
class nnFormerTrainerV2_nnformer_synapse_7(nnFormerTrainer_synapse):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True
        self.load_pretrain_weight=False
        
        self.load_plans_file()    
        
        if len(self.plans['plans_per_stage'])==2:
            Stage=1
        else:
            Stage=0
            
        self.crop_size=self.plans['plans_per_stage'][Stage]['patch_size']
        self.input_channels=self.plans['num_modalities']
        self.num_classes=self.plans['num_classes'] + 1
        self.conv_op=nn.Conv3d
        
        self.embedding_dim=192
        self.depths=[2, 2, 2, 2]
        self.num_heads=[6, 12, 24, 48]
        self.embedding_patch_size=[2,4,4]
        self.window_size=[4,4,8,4]
        self.deep_supervision=True

        ################# SET MOMENTUM SCHEDULER TO UPDATE TEACHER #############################
        self.momentum_schedule = cosine_scheduler(0.996, 1,
                                               self.max_num_epochs, 250*2,warmup_epochs=10)

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            
            self.process_plans(self.plans)

            self.setup_DA_params()
            if self.deep_supervision:
                ################# Here we wrap the loss for deep supervision ############
                # we need to know the number of outputs of the network
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                #mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
                #weights[~mask] = 0
                weights = weights / weights.sum()
                print(weights)
                self.ds_loss_weights = weights
                # now wrap the loss
                self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
                ################# END ###################
            
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +"_stage%d" % self.stage)
            seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
            seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))                         
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales if self.deep_supervision else None,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    seeds_train=seeds_train,
                    seeds_val=seeds_val
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.student_network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
  

        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d


        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        
        self.student_network=nnFormer(crop_size=self.crop_size,
                                embedding_dim=self.embedding_dim,
                                input_channels=self.input_channels,
                                num_classes=self.num_classes,
                                conv_op=self.conv_op,
                                depths=self.depths,
                                num_heads=self.num_heads,
                                patch_size=self.embedding_patch_size,
                                window_size=self.window_size,
                                deep_supervision=self.deep_supervision)
        self.teacher_network=nnFormer(crop_size=self.crop_size,
                                embedding_dim=self.embedding_dim,
                                input_channels=self.input_channels,
                                num_classes=self.num_classes,
                                conv_op=self.conv_op,
                                depths=self.depths,
                                num_heads=self.num_heads,
                                patch_size=self.embedding_patch_size,
                                window_size=self.window_size,
                                deep_supervision=self.deep_supervision)

        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

        self.teacher_network.load_state_dict(self.student_network.state_dict())


        # if self.load_pretrain_weight:
        #     checkpoint = torch.load('/home/xychen/jsguo/weight/gelunorm_former_skip_global_shift.model', map_location='cpu')
        #     self.student_network.load_state_dict(checkpoint)
        #     print('I am using the pre_train weight!!')   
        
     
        if torch.cuda.is_available():
            self.student_network.cuda()
            self.teacher_network.cuda()
        self.student_network.inference_apply_nonlin = softmax_helper
        self.teacher_network.inference_apply_nonlin = softmax_helper


    def initialize_optimizer_and_scheduler(self):
        assert self.student_network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.student_network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        if self.deep_supervision:
            target = target[0]
            output = output[0]
        else:
            target = target
            output = output
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.student_network.do_ds
        self.student_network.do_ds = False

        dt = self.teacher_network.do_ds
        self.teacher_network.do_ds = False

        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.student_network.do_ds = ds
        self.teacher_network.do_ds = dt
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        dt = self.teacher_network.do_ds
        self.teacher_network.do_ds = False

        ds = self.student_network.do_ds
        self.student_network.do_ds = False

        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.teacher_network.do_ds = dt
        self.student_network.do_ds = ds
        return ret

    def run_iteration(self, data_generator,idx, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output_teacher = self.teacher_network(data, mask_flag=False)
                output_student = self.student_network(data,mask_flag=False)
                output_student_masked = self.student_network(data,mask_flag=True)
                del data
                l_student = self.loss(output_student, target)

                l_student_masked = self.loss(output_student_masked, target)

                if do_backprop:
                   
                    l = l_student + l_student_masked + 0.4*nn.MSELoss()(output_teacher[0],output_student_masked[0])/ torch.linalg.norm(output_teacher[0])    #0.5*entropy_loss(output_teacher[0],output_student[0], num_classes=self.num_classes)   #l_student + l_teacher +  #1.0*sum([dist_loss(t,s) for t,s in zip(output_teacher,output_student)])/3.0   #dist_loss(output_teacher[0],output_student[0])  #    #0.5*sum([self.consistency_loss(t,s) for t,s in zip(output_student,output_teacher)])
                    
                else:
                    l = l_student

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output_teacher = self.teacher_network(data, mask_flag=False)
            if do_backprop:
                output_student = self.student_network(data,mask_flag=False)
                output_student_masked = self.student_network(data,mask_flag=True)
            else:
                output_student = self.student_network(data,mask_flag=False)
            del data
            l_student = self.loss(output_student, target)
            l_teacher = self.loss(output_teacher, target)

            if do_backprop:
                l_student_masked = self.loss(output_student_masked, target)
                l = l_student + l_teacher + l_student_masked  + 0.25*nn.MSELoss()(output_teacher[0],output_student_masked[0])/ torch.linalg.norm(output_teacher[0])     #0.5*entropy_loss(output_teacher[0],output_student[0], num_classes=self.num_classes)    #  l_student + l_teacher #0.5*dist_loss(output_teacher[0],output_student[0])    #1.0*sum([dist_loss(t,s) for t,s in zip(output_teacher,output_student)])/3.0   #dist_loss(output_teacher[0],output_student[0])  #    #0.5*sum([self.consistency_loss(t,s) for t,s in zip(output_student,output_teacher)])
            else:
                l = l_teacher + l_student

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.student_network.parameters(), 12)
                self.optimizer.step()

         ##### EMA update for teacher #######
        if do_backprop:
            with torch.no_grad():
                m =  self.momentum_schedule[idx]  # momentum parameter
                for param_q, param_k in zip(self.student_network.parameters(), self.teacher_network.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


        if run_online_evaluation:
            self.run_online_evaluation(output_student, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            splits[self.fold]['train']=np.array(['img0006','img0007' ,'img0009', 'img0010', 'img0021' ,'img0023' ,'img0024','img0026' ,'img0027' ,'img0031', 'img0033' ,'img0034' \
                                ,'img0039', 'img0040','img0005', 'img0028', 'img0030', 'img0037'])
            #splits[self.fold]['train']=np.array(['img0006','img0007' ,'img0009', 'img0010', 'img0021'])
            # splits[self.fold]['train']=np.array(['img0006'])
            splits[self.fold]['val']=np.array(['img0001', 'img0002', 'img0003', 'img0004', 'img0008', 'img0022','img0025', 'img0029', 'img0032', 'img0035', 'img0036', 'img0038'])
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.student_network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.student_network.do_ds
        dt = self.teacher_network.do_ds
        if self.deep_supervision:
            self.student_network.do_ds = True
            self.teacher_network.do_ds = True
        else:
            self.student_network.do_ds = False
            self.teacher_network.do_ds = False
        ret = super().run_training()
        self.student_network.do_ds = ds
        self.teacher_network.do_ds = dt
        return ret
