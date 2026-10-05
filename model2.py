# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
# !pip install gputil
# !pip install psutil
# !pip install humanize


import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()

gpu = GPUs[0]
def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

#!wget https://archive.org/download/documerica/documerica.tar.gz

#!wget https://archive.org/download/documerica/dustified_64.tar.gz

#!tar -xvzf documerica.tar.gz -C '/content/drive/My Drive/documerica/'

#!tar -xvzf dustified_64.tar.gz -C '/content/drive/My Drive/documerica/documerica/'

#!pip install wandb

import fastai
import wandb
from wandb.fastai import WandbCallback
from fastai.vision import *
from fastai.callbacks import *
from fastai.callbacks.hooks import *
from fastai.vision.gan import *
from functools import partial
import matplotlib.pyplot as plt
from skimage import io, color, exposure
import numpy as np
import scipy.ndimage
import cv2
from matplotlib import cm
import pytorch_ssim
from torchvision.models import resnet34, vgg16_bn

%matplotlib inline

torch.manual_seed(420)
torch.cuda.manual_seed_all(420)
np.random.seed(420)
dtype = torch.float

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

project_id = 'ScratchRemovalFeatureLoss4'
experiment_id = project_id+'-2'

path = '/home/jupyter/documerica'
path_hr = path+'/dataset/clean'
path_hr_256 = path+'/dataset/clean_256'
path_hr_128 = path+'/dataset/clean_128'
path_hr_64 = path+'/dataset/clean_64'
path_lr = path+'/dataset/dustified_512'
path_lr_256 = path+'/dataset/dustified_256'
path_lr_128 = path+'/dataset/dustified_128'
path_lr_64 = path+'/dataset/dustified_64'

il = ImageList.from_folder(path_hr)
il

# def resize_one(fn, i, path, size):
#     dest = path/fn.relative_to(path_hr)
#     dest.parent.mkdir(parents=True, exist_ok=True)
#     img = PIL.Image.open(fn)
#     targ_sz = resize_to(img, size, use_min=True)
#     img = img.resize(targ_sz, resample=PIL.Image.ANTIALIAS).convert('RGB')
#     img.save(dest, quality=100)

# # create smaller image sets the first time this nb is run
# sets = [(path_hr_256, 256), (path_hr_128, 128), (path_hr_64, 64)]
# for p,size in sets:
#     if not os.path.exists(p):
#         print(f"resizing to {size} into {p}")
#         parallel(partial(resize_one, path=p, size=size), il.items)

tfms_gen = get_transforms(max_zoom=1.5, max_lighting=0.5, max_warp=0.25)
tfms_gen

data_images = []
for file in sorted(os.listdir(path_hr)):
    if file[-4:] == '.png': data_images.append(file)

df_images = pd.DataFrame(data_images, columns=['Filename'])
df_images

#train, validate, test = np.split(df_images.sample(random_state=np.random.RandomState(42), frac=1), [int(.8*len(df_images)), int(.9*len(df_images))])

# os.mkdir(path+'/split/')
# train.to_csv(path+'/split/train.csv')
# test.to_csv(path+'/split/test.csv')
# validate.to_csv(path+'/split/validate.csv')

train = pd.read_csv(path+'/split/train.csv', index_col=0)
test = pd.read_csv(path+'/split/test.csv', index_col=0)
validate = pd.read_csv(path+'/split/validate.csv', index_col=0)

train

validate

test

src = ImageImageList.from_df(df=df_images, path=path_lr, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=validate.index.to_list())
src

src_256 = ImageImageList.from_df(df=df_images, path=path_lr_256, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=validate.index.to_list())
src_256

src_128 = ImageImageList.from_df(df=df_images, path=path_lr_128, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=validate.index.to_list())
src_128

src_64 = ImageImageList.from_df(df=df_images, path=path_lr_64, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=validate.index.to_list())
src_64

src.train[97]

src_256.train[97]

src_128.train[97]

src_64.train[97]

def plots_f(img, rows, cols, width, height, **kwargs):
    [img.apply_tfms(tfms_gen[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
        rows,cols,figsize=(width,height))[1].flatten())]

plots_f(src_256.train[1133], 2, 4, 20, 10)

plots_f(src_256.train[100], 2, 4, 20, 10)

src_test = ImageImageList.from_df(df=df_images, path=path_lr, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=test.index.to_list())
src_test

src_test_256 = ImageImageList.from_df(df=df_images, path=path_lr_256, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=test.index.to_list())
src_test_256

src_test_128 = ImageImageList.from_df(df=df_images, path=path_lr_128, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=test.index.to_list())
src_test_128

src_test_64 = ImageImageList.from_df(df=df_images, path=path_lr_64, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=test.index.to_list())
src_test_64

def get_data(src, bs, size, tfms, path_hr):
    data = (src.label_from_func(lambda x: '{}/{}'.format(path_hr,os.path.basename(x))) # Ensure that we get the matching clean image for each dusty one
           .transform(tfms, size=size, tfm_y=True)
           .databunch(bs=bs).normalize(do_y=True))
    data.c = 3
    return data

def show_xyzdiffs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets), `zs` (predictions) and `diffs` (differences)  on a figure of `figsize`."
        title = 'Input / Prediction / Target / Difference'
        axs = subplots(len(xs), 4, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], **kwargs)
            y.show(ax=axs[i,2], **kwargs)
            z.show(ax=axs[i,1], **kwargs)
            np_diff = np.subtract(z.data.numpy(), y.data.numpy()).astype(np.float32)
            np_diff = (np_diff - np.min(np_diff))/np.ptp(np_diff)
            diff = Image(torch.from_numpy(np_diff).to(device))
            diff.show(ax=axs[i,3], **kwargs)

def _show_results(self, ds_type=DatasetType.Valid, rows:int=5, **kwargs):
        "Show `rows` result of predictions on `ds_type` dataset."
        #TODO: get read of has_arg x and split_kwargs_by_func if possible
        #TODO: simplify this and refactor with pred_batch(...reconstruct=True)
        n_items = rows ** 2 if self.data.train_ds.x._square_show_res else rows
        if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
        ds = self.dl(ds_type).dataset
        self.callbacks.append(RecordOnCPU())
        preds = self.pred_batch(ds_type)
        *self.callbacks,rec_cpu = self.callbacks
        x,y = rec_cpu.input,rec_cpu.target
        norm = getattr(self.data,'norm',False)
        if norm:
            x = self.data.denorm(x)
            if norm.keywords.get('do_y',False):
                y     = self.data.denorm(y, do_x=True)
                preds = self.data.denorm(preds, do_x=True)
        analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
        preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(n_items)]
        xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        if has_arg(ds.y.reconstruct, 'x'):
            ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
            zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
        else :
            ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
            zs = [ds.y.reconstruct(z) for z in preds]
        show_xyzdiffs(ds.x, xs, ys, zs, **kwargs)

arch = models.resnet34
wd = 1e-03 # weight decay
y_range = (-3.,3.)

def ssim_metric(inputs, targets, window_size=7, size_average = True):
    ssim_metric = pytorch_ssim.ssim(inputs, targets, window_size, size_average)
    return ssim_metric

class SSIM_metric(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, func):
        # If it's a partial, use func.func
        name = getattr(func,'func',func).__name__
        self.func, self.name = func, name

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        self.count += last_target[0].size(0)
        val = self.func(last_output, *last_target)
        self.val += last_target[0].size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]

blocks

class FeatureLoss(nn.Module):
    def __init__(self, base_loss, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = base_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['MAE'] + [f'feat_L1_{i}' for i in range(len(layer_ids))
              ] + [f'feat_L1_gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input,target)]
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [self.base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

loss_gen = FeatureLoss(F.l1_loss, vgg_m, blocks, [0, 0, 20, 70, 10])

data_gen = get_data(src_64, 32, 64, tfms_gen, path_hr_64)

data_gen

data_gen.show_batch()

wandb.login()

wandb.init(project=project_id, tags=['generator', 'pre-train', 'stage1', experiment_id])

generator = unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen, path=path, metrics=[mean_absolute_error, mean_squared_error, root_mean_squared_error, SSIM_metric(ssim_metric)]).to_fp16()

generator.summary()

generator.model

generator.lr_find()
generator.recorder.plot()

wandb.config.update({"batch size": 32, 
                     "image size": 64,
                     "backbone": "resnet34",
                     "y_range": (-3.,3.),
                     "pct_start": 0.8,
                     "weight_decay": 1e-03,
                     "max_lr": 1e-03,
                     "loss": 'FeatureLossResNet',
                     "norm_type": NormType.Weight,
                     "self-attention": True,
                     "blur": True,
                     "frozen": True,
                     "half-precision": True}, allow_val_change=True)

generator.callback_fns.append(partial(WandbCallback, input_type='images', log='all'))

generator.callback_fns.append(partial(LossMetrics))

generator.fit_one_cycle(10, pct_start=0.8, max_lr=1e-03)

generator.recorder.plot_lr()

generator.recorder.plot_losses()

generator.show_results = _show_results

generator.show_results(generator, rows=10)

generator.save('gen-pre-64-frozen-'+experiment_id)

del generator
del data_gen

data_gen = get_data(src_128, 16, 128, tfms_gen, path_hr_128)

data_gen

data_gen.show_batch(imgsize=5)

wandb.init(project=project_id, tags=['generator', 'pre-train', 'stage2', experiment_id])

generator = unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen, path=path, metrics=[mean_absolute_error, mean_squared_error, root_mean_squared_error, SSIM_metric(ssim_metric)]).load('gen-pre-64-frozen-'+experiment_id).to_fp16()

generator.summary()

generator.model

generator.layer_groups

wandb.config.update({"batch size": 16, 
                     "image size": 128,
                     "backbone": "resnet34",
                     "y_range": (-3.,3.),
                     "pct_start": 0.8,
                     "weight_decay": 1e-03,
                     "max_lr": 1e-03,
                     "loss": 'FeaturLossResNet',
                     "norm_type": NormType.Weight,
                     "self-attention": True,
                     "blur": True,
                     "frozen": True,
                     "half-precision": True}, allow_val_change=True)

generator.callback_fns.append(partial(WandbCallback, input_type='images', log='all'))

generator.callback_fns.append(partial(LossMetrics))

generator.fit_one_cycle(10, pct_start=0.8, max_lr=1e-03)

generator.show_results = _show_results

generator.show_results(generator, rows=10, figsize=(20,60))

generator.save('gen-pre-128-frozen-'+experiment_id)

del generator
del data_gen

data_gen = get_data(src_256, 8, 256, tfms_gen, path_hr_256)

data_gen.show_batch(imgsize=5)

generator = unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen, path=path, metrics=[mean_absolute_error, mean_squared_error, root_mean_squared_error, SSIM_metric(ssim_metric)]).load('gen-pre-128-frozen-'+experiment_id).to_fp16()

generator.summary()

generator.model

generator.layer_groups

wandb.init(project=project_id, tags=['generator', 'pre-train', experiment_id])

wandb.config.update({"batch size": 8, 
                     "image size": 256,
                     "backbone": "resnet34",
                     "y_range": (-3.,3.),
                     "pct_start": 0.8,
                     "weight_decay": 1e-03,
                     "max_lr": 1e-03,
                     "loss": 'FeatureLossResNet',
                     "norm_type": NormType.Weight,
                     "self-attention": True,
                     "blur": True,
                     "frozen": True,
                     "half-precision": True}, allow_val_change=True)

generator.callback_fns.append(partial(WandbCallback, input_type='images', log='all'))

generator.callback_fns.append(partial(LossMetrics))

generator.fit_one_cycle(10, pct_start=0.8, max_lr=1e-03)

generator.show_results = _show_results

generator.show_results(generator, rows=10, figsize=(30,80))

generator.save('gen-pre-256-frozen-'+experiment_id)

del generator
del data_gen

data_gen = get_data(src_256, 8, 256, tfms_gen, path_hr_256)

data_gen.show_batch(imgsize=5)

generator = unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen, path=path, metrics=[mean_absolute_error, mean_squared_error, root_mean_squared_error, SSIM_metric(ssim_metric)]).load('gen-pre-256-frozen-'+experiment_id).to_fp16()

generator.unfreeze()

generator.summary()

generator.model

generator.layer_groups

generator.lr_find()
generator.recorder.plot()

wandb.init(project=project_id, tags=['generator', 'pre-train', experiment_id])

wandb.config.update({"batch size": 8, 
                     "image size": 256,
                     "backbone": "resnet34",
                     "y_range": (-3.,3.),
                     "pct_start": 0.5,
                     "weight_decay": 1e-03,
                     "max_lr": (1e-05, 1e-03),
                     "loss": 'FeatureLossResNet',
                     "norm_type": NormType.Weight,
                     "self-attention": True,
                     "blur": True,
                     "frozen": False,
                     "half-precision": True}, allow_val_change=True)

generator.callback_fns.append(partial(WandbCallback, input_type='images', log='all'))

generator.callback_fns.append(partial(LossMetrics))

generator.fit_one_cycle(20, pct_start=0.5, max_lr=slice(1e-05,1e-03))

generator.show_results = _show_results

generator.show_results(generator, rows=10, figsize=(30,80))

generator.save('gen-pre-256-unfrozen-'+experiment_id)

del generator
del data_gen

src_test_256_clean = ImageImageList.from_df(df=df_images, path=path_hr_256, cols=['Filename']).split_by_idxs(train_idx=train.index.to_list(), valid_idx=test.index.to_list())
src_test_256_clean

tfms_test = get_transforms(do_flip=False, max_zoom=1., max_lighting=None, max_rotate=None, max_warp=None)

data_gen_test = get_data(src_test_256, 1, 256, tfms_test, path_hr_256)

data_gen_test_clean = get_data(src_test_256_clean, 1, 256, tfms_test, path_hr_256)

data_gen_test.show_batch()

data_gen_test_clean.show_batch()

def ssim_eval(inputs, targets, window_size=3, size_average = True, **kwargs):
    ssim = pytorch_ssim.ssim(inputs, targets, window_size, size_average)
    return ssim
    

class SSIM_metric_eval(nn.Module):
    def __init__(self, window_size=7, size_average=True, per_channel=False):
        super().__init__()
        self.size_average = size_average
        self.window_size = window_size
        self.per_channel = per_channel
        
    def forward(self, input, target, **kwargs):
        ssim_per_chan = []
        ssim_per_chan.append(ssim_eval(input, target, window_size=self.window_size, size_average=self.size_average, **kwargs))
        if self.per_channel:
            input_channels = torch.chunk(input,3,1)
            target_channels = torch.chunk(target,3,1)
            for i in range(3):
                ssim_per_chan.append(ssim_eval(input_channels[i], target_channels[i], window_size=self.window_size, size_average=self.size_average, **kwargs))
        return torch.Tensor(ssim_per_chan)

def ssim_init(data1, data2):
    ssim_dusty_clean = []
    ssim_clean_clean = []
    for i in iter(data1.valid_dl):
        ssim_dusty_clean.append(ssim_eval(i[0],i[1]).item())
    
    for i in iter(data2.valid_dl):
        ssim_clean_clean.append(ssim_eval(i[0],i[1]).item())
        
    return pd.DataFrame({'SSIM(clean, clean)': ssim_clean_clean, 'SSIM(dusty, clean)': ssim_dusty_clean})

ssim_data = ssim_init(data_gen_test, data_gen_test_clean)

ssim_data

from matplotlib.lines import Line2D

def plot_ssim(df, n_rows, n_cols, title, x_label, y_label, colours, figsize=(15,5)):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, c in enumerate(df.columns):
        curr_ax = df[c].plot(kind='hist', ax=axes.flatten()[i], figsize=figsize, title=c, bins=100, color=colours[i], range=(0.0, 1.0))
        if df[c].mean() != 1:
            curr_ax.axvline(df[c].mean(), color='k', linestyle='dashed', linewidth=1)
            curr_ax.axvline(df[c].mean()+df[c].std(), color='k', linestyle='dotted', linewidth=1)
            curr_ax.axvline(df[c].mean()-df[c].std(), color='k', linestyle='dotted', linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            #curr_ax.text(df[c].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(df[c].mean()))
            #curr_ax.text(df[c].mean()*1.1, max_ylim*0.8, 'SD: {:.2f}'.format(df[c].std()))
            lines = [Line2D([0], [0], color='k', linewidth=3, linestyle='dashed'), Line2D([0], [0], color='k', linestyle='dotted')]
            labels = ['Mean: {:.2f}'.format(df[c].mean()), 'SD: {:.2f}'.format(df[c].std())]
            curr_ax.legend(lines, labels, loc="upper left")
    plt.suptitle(title, x=0.5, y=1.05, ha='center', fontsize='x-large')
    for ax in axes.flatten():
        ax.grid(False)
    fig.text(0.5, 0.001, x_label, ha='center', fontsize='larger')

def plot_delta(df, n_rows, n_cols, title, x_label, y_label, colours, figsize=(15,5)):
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i, c in enumerate(df.columns):
        curr_ax = df[c].plot(kind='hist', ax=axes, figsize=figsize, title=c, bins=100, color=colours[i], range=(-0.5, 0.5))
        if df[c].mean() != 1:
            curr_ax.axvline(df[c].mean(), color='k', linestyle='dashed', linewidth=1)
            curr_ax.axvline(df[c].mean()+df[c].std(), color='k', linestyle='dotted', linewidth=1)
            curr_ax.axvline(df[c].mean()-df[c].std(), color='k', linestyle='dotted', linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            #curr_ax.text(df[c].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(df[c].mean()))
            #curr_ax.text(df[c].mean()*1.1, max_ylim*0.8, 'SD: {:.2f}'.format(df[c].std()))
            lines = [Line2D([0], [0], color='k', linewidth=3, linestyle='dashed'), Line2D([0], [0], color='k', linestyle='dotted')]
            labels = ['Mean: {:.2f}'.format(df[c].mean()), 'SD: {:.2f}'.format(df[c].std())]
            curr_ax.legend(lines, labels, loc="upper left")
    plt.suptitle(title, x=0.5, y=1.05, ha='center', fontsize='x-large')
    axes.grid(False)
    fig.text(0.5, 0.001, x_label, ha='center', fontsize='larger')

plot_ssim(ssim_data, 1, 2, 'Distribution of SSIM scores between (clean, clean) and (dusty, clean) images', 'SSIM Score', 'Frequency', ['blue', 'blue'])

def validate_ssim(model, metric, data, average=False):
    cb_handler = CallbackHandler(model.callbacks + [], None)
    metric = fastai.basic_train.validate(
        model.model, 
        data.valid_dl, 
        metric,
        cb_handler,
        average=average)
    ssims = torch.stack(metric).tolist()
    return pd.DataFrame(ssims, columns=['SSIM(pred, clean)','SSIM(pred, clean) channel 1', 'SSIM(pred, clean) channel 2', 'SSIM(pred, clean) channel 3'])

generator_test = unet_learner(data_gen_test, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen, path=path).load('gen-pre-256-unfrozen-'+experiment_id).to_fp16()

ssim_dusty_clean = validate_ssim(generator_test, SSIM_metric_eval(per_channel=True), data_gen_test)

preds = generator_test.get_preds()

def _show_results_eval(self, ds_type=DatasetType.Valid, rows:int=1, indexes=[], **kwargs):
        "Show `rows` result of predictions on `ds_type` dataset."
        #TODO: get read of has_arg x and split_kwargs_by_func if possible
        #TODO: simplify this and refactor with pred_batch(...reconstruct=True)
        n_items = rows ** 2 if self.data.train_ds.x._square_show_res else rows
        if indexes != []:
            for batch_index, batch_data in enumerate(self.dl(ds_type)):
                if batch_index in indexes: 
                    ds = self.dl(ds_type).dataset
                    self.callbacks.append(RecordOnCPU())
                    preds = self.pred_batch(ds_type, batch=batch_data)
                    *self.callbacks,rec_cpu = self.callbacks
                    x,y = rec_cpu.input,rec_cpu.target
                    norm = getattr(self.data,'norm',False)
                    if norm:
                        x = self.data.denorm(x)
                        if norm.keywords.get('do_y',False):
                            y     = self.data.denorm(y, do_x=True)
                            preds = self.data.denorm(preds, do_x=True)
                    analyze_kwargs,kwargs = split_kwargs_by_func(kwargs, ds.y.analyze_pred)
                    preds = [ds.y.analyze_pred(grab_idx(preds, i), **analyze_kwargs) for i in range(n_items)]
                    xs = [ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
                    if has_arg(ds.y.reconstruct, 'x'):
                        ys = [ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
                        zs = [ds.y.reconstruct(z, x=x) for z,x in zip(preds,xs)]
                    else :
                        ys = [ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
                        zs = [ds.y.reconstruct(z) for z in preds]
                    show_xyzdiffs(ds.x, xs, ys, zs, **kwargs)

ssim_dusty_clean

plot_ssim(ssim_dusty_clean, 2, 2, 'Distribution of SSIM scores between predictions on dusty images and corresponding clean images', 'SSIM Score', 'Frequency', ['orange', 'red', 'green', 'blue'], (10,7))

ssim_dusty_clean = ssim_dusty_clean.join(ssim_data['SSIM(dusty, clean)'])
ssim_dusty_clean

ssim_dusty_clean['Δ(SSIM(pred, clean), SSIM(dusty, clean))'] = ssim_dusty_clean['SSIM(pred, clean)'] - ssim_dusty_clean['SSIM(dusty, clean)']
ssim_dusty_clean

plot_delta(ssim_dusty_clean[['Δ(SSIM(pred, clean), SSIM(dusty, clean))']], 1, 1, 'Distribution of improvement in SSIM scores after network prediction', 'SSIM Score', 'Frequency', ['blue'], figsize=(10,3))

ssim_dusty_clean.nlargest(3, columns=['Δ(SSIM(pred, clean), SSIM(dusty, clean))'])

indexes = ssim_dusty_clean.nlargest(3, columns=['Δ(SSIM(pred, clean), SSIM(dusty, clean))']).index.to_list()

generator_test.show_results = _show_results_eval(generator_test, indexes=indexes, imgsize=9)

ssim_dusty_clean.nsmallest(3, columns=['Δ(SSIM(pred, clean), SSIM(dusty, clean))'])

indexes = ssim_dusty_clean.nsmallest(3, columns=['Δ(SSIM(pred, clean), SSIM(dusty, clean))']).index.to_list()
indexes

generator_test.show_results = _show_results_eval(generator_test, indexes=indexes, imgsize=7)

generator_test = unet_learner(data_gen_test_clean, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen, path=path).load('gen-pre-256-unfrozen-'+experiment_id).to_fp16()

ssim_clean_clean = validate_ssim(generator_test, SSIM_metric_eval(per_channel=True), data_gen_test_clean)

plot_ssim(ssim_clean_clean, 2, 2, 'Distribution of SSIM scores between predictions on clean images and corresponding clean images', 'SSIM Score', 'Frequency', ['orange', 'red', 'green', 'blue'], (10,7))

ssim_clean_clean

ssim_clean_clean.nsmallest(3, columns=['SSIM(pred, clean)'])

indexes = ssim_clean_clean.nsmallest(3, columns=['SSIM(pred, clean)']).index.to_list()

generator_test.show_results = _show_results_eval(generator_test, indexes=indexes, imgsize=7)

ssim_clean_clean.nlargest(3, columns=['SSIM(pred, clean)'])

indexes = ssim_clean_clean.nlargest(3, columns=['SSIM(pred, clean)']).index.to_list()

generator_test.show_results = _show_results_eval(generator_test, indexes=indexes, imgsize=7)

