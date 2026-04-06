import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.utils.data
from metric import psnr, ssim
from data import RESIDE_Dataset, TestDataset
from option.Base_DEA import opt
from DEAmodel import DEANet
from loss import loss

start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(model, loader_train, loader_test, Syn_optim):
    losses = []
    loss_log = {'Loss': []}
    loss_log_tmp = {'Loss': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    loader_train_iter_1 = iter(loader_train)

    for step in range(start_step + 1, steps + 1):
        model.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in Syn_optim.param_groups:
                param_group["lr"] = lr

        x, y = next(loader_train_iter_1)
        x = x.to(opt.device)
        y = y.to(opt.device)

        Syn_optim.zero_grad()

        out = model(x)

        if opt.w_loss_l1 > 0:
            l1_loss = criterion[0](out, y)

        loss = opt.w_loss_l1 * l1_loss

        loss.backward()

        Syn_optim.step()

        losses.append(loss.item())

        loss_log_tmp['Loss'].append(loss.item())

        print(
            f'\r| Loss:{loss.item():.5f} | step :{step}/{steps} | lr :{lr :.9f}  | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        if step % len(loader_train) == 0:
            loader_train_iter_1 = iter(loader_train)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)
        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or (
                step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train)) == 0):
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (
                        5 * len(loader_train))
            else:
                epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = test(model, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f} | lr:{lr:.12f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            state_dict = model.state_dict()
            if 'module' in list(state_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict
            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(
                    f'model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                torch.save(state_dict, saved_best_model_path)
            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pth')
            torch.save(state_dict, saved_single_model_path)
            loader_train_iter_1 = iter(loader_train)
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W]
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_data_dir(candidates, required_subdirs):
    for candidate in candidates:
        if os.path.isdir(candidate) and all(os.path.isdir(os.path.join(candidate, s)) for s in required_subdirs):
            return candidate
    raise FileNotFoundError(
        f"Cannot find dataset directory. Tried: {candidates}. Required subfolders: {required_subdirs}"
    )


if __name__ == "__main__":
    set_seed_torch(3407)

    model = DEANet().to(opt.device)

    project_root = os.path.dirname(os.path.abspath(__file__))

    train_dir_1 = resolve_data_dir(
        [
            os.path.join(project_root, 'dataset', 'Haze4K', 'train'),
            os.path.join(project_root, 'dataset', 'Synthetic_Data', 'train'),
        ],
        required_subdirs=('hazy', 'clear')
    )
    train_set_1 = RESIDE_Dataset(train_dir_1, True, 256, '.png')

    test_dir = resolve_data_dir(
        [
            os.path.join(project_root, 'dataset', 'Haze4K', 'val'),
            os.path.join(project_root, 'dataset', 'Haze4K', 'test'),
            os.path.join(project_root, 'dataset', 'Synthetic_Data', 'test'),
        ],
        required_subdirs=('hazy', 'clear')
    )
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'), '.png')

    loader_train = DataLoader(dataset=train_set_1, batch_size=16, shuffle=True, num_workers=8)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)

    criterion = []
    criterion.append(loss.LossFunction().to(opt.device))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=opt.start_lr,
                           betas=opt.betas, eps=opt.eps)

    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))
    print("------------------------------------------------------------------")

    optimizer.zero_grad()
    train(model, loader_train, loader_test, optimizer)
