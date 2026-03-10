import math
import os
import time
import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.utils.data
from metric import psnr, ssim
from data import TestDataset, CLIP_loader
from option.BiLaLoRA_DEA import opt
from DEAmodel import Backbonecs
import torchvision.transforms as TT

clip_preprocessor = TT.Compose([
    TT.Resize((224, 224), interpolation=TT.InterpolationMode.BICUBIC, antialias=True),
    TT.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def directional_clip_loss(output_images, input_images, clip_model, text_features):

    processed_output = clip_preprocessor(output_images)
    processed_input = clip_preprocessor(input_images)

    E_out = clip_model.encode_image(processed_output)
    E_in = clip_model.encode_image(processed_input)

    E_out_norm = F.normalize(E_out, p=2, dim=-1)
    E_in_norm = F.normalize(E_in, p=2, dim=-1)

    V_neg_norm = F.normalize(text_features[0:1], p=2, dim=-1)
    V_pos_norm = F.normalize(text_features[1:2], p=2, dim=-1)

    delta_image = E_out_norm - E_in_norm
    delta_text = V_pos_norm - V_neg_norm

    delta_image_norm = F.normalize(delta_image, p=2, dim=-1)
    delta_text_norm = F.normalize(delta_text, p=2, dim=-1)

    cosine_similarity = (delta_image_norm @ delta_text_norm.T).squeeze()

    loss = 1.0 - cosine_similarity

    return loss.mean()


class Architect(object):

    def __init__(self, model, submodule_alphas, opt_config=None):
        self.model = model
        self.submodule_alphas = submodule_alphas

        if opt_config is None:
            opt_config = {
                "arch_learning_rate": 3e-4,
                "arch_weight_decay": 1e-3,
                "momentum": 0.9,
                "weight_decay": 3e-4,
            }

        self.network_momentum = opt_config.get("momentum", 0.9)
        self.network_weight_decay = opt_config.get("weight_decay", 3e-4)


        self.optimizer = torch.optim.AdamW(
            list(submodule_alphas.values()),
            lr=opt_config.get("arch_learning_rate", 3e-4),
            betas=(0.5, 0.999),
            weight_decay=opt_config.get("arch_weight_decay", 1e-3)
        )

    def step(self, input_train, target_train, input_valid, target_valid,
             eta, network_optimizer, unrolled=True):

        self.optimizer.zero_grad()

        if unrolled:
            self._backward_step_unrolled(
                input_train, input_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid)

        self.optimizer.step()

    def _backward_step(self, input_valid):

        pred = self.model(input_valid)
        loss = directional_clip_loss(
            output_images=pred,
            input_images=input_valid,
            clip_model=clip_model,
            text_features=text_features
        )
        loss = opt.w_loss_H2C * loss


        loss.backward()

    def _backward_step_unrolled(self, input_train, input_valid, eta,
                                network_optimizer):

        pred_valid = self.model(input_valid)
        unrolled_loss = directional_clip_loss(
            output_images=pred_valid,
            input_images=input_valid,
            clip_model=clip_model,
            text_features=text_features
        )
        unrolled_loss = opt.w_loss_H2C * unrolled_loss


        unrolled_loss.backward()


        dalpha = [v.grad.clone() if v.grad is not None else torch.zeros_like(v)
                  for v in self.submodule_alphas.values()]


        lora_params = [p for p in self.model.parameters()
                       if p.requires_grad and 'alpha' not in str(p)]

        vector = []
        for v in lora_params:
            if v.grad is not None:
                vector.append(v.grad.data.clone())
            else:
                vector.append(torch.zeros_like(v))


        self.model.zero_grad()
        pred_train = self.model(input_train)
        lower_loss = directional_clip_loss(
            output_images=pred_train,
            input_images=input_train,
            clip_model=clip_model,
            text_features=text_features
        )
        lower_loss = opt.w_loss_H2C * lower_loss

        dfy = torch.autograd.grad(
            lower_loss,
            lora_params,
            retain_graph=True,
            allow_unused=True
        )

        gfyfy = 0
        gFyfy = 0

        for f, F in zip(dfy, vector):
            if f is None:
                f = torch.zeros_like(F)
            gfyfy = gfyfy + torch.sum(f * f)
            gFyfy = gFyfy + torch.sum(F * f)


        GN_loss = -gFyfy.detach() / (gfyfy.detach() + 1e-8) * lower_loss

        implicit_grads = torch.autograd.grad(
            GN_loss,
            list(self.submodule_alphas.values()),
            allow_unused=True
        )


        for alpha_param, g, ig in zip(self.submodule_alphas.values(), dalpha, implicit_grads):
            if ig is None:
                ig = torch.zeros_like(g)
            final_grad = g - eta * ig.data

            if alpha_param.grad is None:
                alpha_param.grad = Variable(final_grad)
            else:
                alpha_param.grad.data.copy_(final_grad)


class LoRALayer(nn.Module):


    def __init__(self, original_layer: nn.Conv2d, r: int, lora_alpha: int, shared_alpha: nn.Parameter = None):
        super().__init__()


        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size[0]


        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r


        self.lora_A = nn.Parameter(
            torch.zeros(r * kernel_size, in_channels * kernel_size)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels * kernel_size, r * kernel_size)
        )


        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


        if shared_alpha is not None:
            self.alpha = shared_alpha
        else:

            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):

        lora_weight = (self.lora_B @ self.lora_A).view(self.original_layer.weight.shape)
        gate = torch.sigmoid(self.alpha)

        combined_weight = self.original_layer.weight + gate * self.scaling * lora_weight

        return F.conv2d(
            x,
            combined_weight,
            self.original_layer.bias,
            self.original_layer.stride,
            self.original_layer.padding,
            self.original_layer.dilation,
            self.original_layer.groups
        )

    def __repr__(self):
        gate_value = torch.sigmoid(self.alpha).item()
        return (f"LoRALayer(r={self.r}, lora_alpha={self.lora_alpha}, "
                f"alpha={self.alpha.item():.8f}, gate={gate_value:.8f})")


def apply_lora(model: nn.Module, target_submodules: list, r: int, lora_alpha: int):
    print("\n" + "=" * 70)
    print("Applying LoRA layers with shared alpha per submodule...")
    print("=" * 70)


    submodule_alphas = {}

    for submodule_name in target_submodules:
        try:
            parent_module = model.get_submodule(submodule_name)
            print(f"\n-> Processing submodule: '{submodule_name}'")

            shared_alpha = nn.Parameter(torch.zeros(1))
            submodule_alphas[submodule_name] = shared_alpha

            conv_count = 0
            for child_name, child_module in list(parent_module.named_children()):
                if isinstance(child_module, nn.Conv2d):
                    new_layer = LoRALayer(
                        original_layer=child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        shared_alpha=shared_alpha
                    ).to(child_module.weight.device)

                    new_layer.original_layer.weight.data.copy_(child_module.weight.data)
                    if child_module.bias is not None:
                        new_layer.original_layer.bias.data.copy_(child_module.bias.data)

                    setattr(parent_module, child_name, new_layer)
                    conv_count += 1
                    print(f"  ✓ Replaced '{child_name}' with LoRALayer (shared alpha)")

            print(f"  Total Conv2d layers replaced: {conv_count}")

        except AttributeError:
            print(f"  ✗ Warning: Submodule '{submodule_name}' not found!")

    print("\n" + "=" * 70)
    print(f"LoRA application completed! Created {len(submodule_alphas)} shared alpha parameters")
    print("=" * 70 + "\n")

    return model, submodule_alphas


def apply_lora_selected_only(model: nn.Module, selected_submodules: list, r: int, lora_alpha: int):
    print("\n" + "=" * 70)
    print(f"Applying LoRA ONLY to selected {len(selected_submodules)} submodules...")
    print("=" * 70)

    submodule_alphas = {}

    for submodule_name in selected_submodules:
        try:
            parent_module = model.get_submodule(submodule_name)
            print(f"\n-> Processing selected submodule: '{submodule_name}'")

            shared_alpha = nn.Parameter(torch.zeros(1))
            submodule_alphas[submodule_name] = shared_alpha

            conv_count = 0
            for child_name, child_module in list(parent_module.named_children()):
                if isinstance(child_module, nn.Conv2d):
                    new_layer = LoRALayer(
                        original_layer=child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        shared_alpha=shared_alpha
                    ).to(child_module.weight.device)

                    new_layer.original_layer.weight.data.copy_(child_module.weight.data)
                    if child_module.bias is not None:
                        new_layer.original_layer.bias.data.copy_(child_module.bias.data)

                    setattr(parent_module, child_name, new_layer)
                    conv_count += 1
                    print(f"  ✓ Replaced '{child_name}' with LoRALayer")

            print(f"  Total Conv2d layers replaced: {conv_count}")

        except AttributeError:
            print(f"  ✗ Warning: Submodule '{submodule_name}' not found!")

    print("\n" + "=" * 70)
    print(f"Selected LoRA application completed! {len(submodule_alphas)} submodules enabled")
    print("=" * 70 + "\n")

    return model, submodule_alphas


def log_architecture_weights(model: nn.Module, submodule_alphas: dict, epoch: int = None):

    header = f"\n{'=' * 70}\n"
    if epoch is not None:
        header += f"Architecture Weights at Epoch {epoch}\n"
    else:
        header += "Current Architecture Weights\n"
    header += "=" * 70
    print(header)

    architecture_params = {}
    submodule_info = []

    for submodule_name, alpha_param in submodule_alphas.items():
        gate_value = torch.sigmoid(alpha_param).item()
        alpha_value = alpha_param.item()

        architecture_params[submodule_name] = {
            'alpha': alpha_value,
            'gate': gate_value
        }

        submodule_info.append({
            'name': submodule_name,
            'alpha': alpha_value,
            'gate': gate_value
        })

    submodule_info_sorted = sorted(submodule_info, key=lambda x: x['gate'], reverse=True)

    print(f"{'Rank':<6} | {'Submodule':<40} | {'Alpha':>12} | {'Gate':>12}")
    print("-" * 76)

    for rank, info in enumerate(submodule_info_sorted, start=1):
        print(f"{rank:<6} | {info['name']:<40} | {info['alpha']:>12.8f} | {info['gate']:>12.8f}")

    print("=" * 70)

    return architecture_params


def get_lora_state_dict(model: nn.Module) -> dict:
    trainable_param_names = {
        name for name, param in model.named_parameters()
        if param.requires_grad
    }

    lora_state_dict = {
        name: param
        for name, param in model.state_dict().items()
        if name in trainable_param_names
    }

    return lora_state_dict


def get_top_k_lora_state_dict(model: nn.Module, submodule_alphas: dict, top_k: int) -> dict:
    submodule_gates = []
    for submodule_name, alpha_param in submodule_alphas.items():
        gate_value = torch.sigmoid(alpha_param).item()
        submodule_gates.append((submodule_name, gate_value, alpha_param))

    submodule_gates_sorted = sorted(submodule_gates, key=lambda x: x[1], reverse=True)

    top_k_submodules = set([item[0] for item in submodule_gates_sorted[:top_k]])

    print(f"\n{'=' * 70}")
    print(f"Top-{top_k} LoRA Submodules (by gate value):")
    print("=" * 70)
    for rank, (name, gate_val, alpha_param) in enumerate(submodule_gates_sorted[:top_k], 1):
        print(f"{rank}. {name}: gate={gate_val:.8f}, alpha={alpha_param.item():.8f}")
    print("=" * 70 + "\n")

    top_k_state_dict = {}
    trainable_param_names = {
        name for name, param in model.named_parameters()
        if param.requires_grad
    }

    for name in trainable_param_names:
        belongs_to_top_k = any(
            submodule_name in name for submodule_name in top_k_submodules
        )

        if belongs_to_top_k:
            top_k_state_dict[name] = model.state_dict()[name]

    return top_k_state_dict, top_k_submodules


def rebuild_model_with_selected_lora(base_model, selected_submodules: list,
                                     lora_state_dict: dict, r: int, lora_alpha: int):

    print("\n" + "=" * 70)
    print("Rebuilding model with ONLY selected LoRA layers...")
    print("=" * 70)
    print(f"Selected submodules: {selected_submodules}")


    for param in base_model.parameters():
        param.requires_grad = False


    new_model, new_submodule_alphas = apply_lora_selected_only(
        model=base_model,
        selected_submodules=selected_submodules,
        r=r,
        lora_alpha=lora_alpha
    )


    print("\nLoading LoRA weights from checkpoint...")
    missing_keys = []
    unexpected_keys = []

    model_state_dict = new_model.state_dict()

    for key, value in lora_state_dict.items():
        if key in model_state_dict:
            model_state_dict[key].copy_(value)
            print(f"  ✓ Loaded: {key}")
        else:
            unexpected_keys.append(key)


    for key in model_state_dict.keys():
        if 'lora' in key or 'alpha' in key:
            if key not in lora_state_dict:
                missing_keys.append(key)

    if missing_keys:
        print(f"\n⚠ Warning: Missing keys in checkpoint:")
        for key in missing_keys:
            print(f"    - {key}")

    if unexpected_keys:
        print(f"\n⚠ Warning: Unexpected keys in checkpoint (will be ignored):")
        for key in unexpected_keys:
            print(f"    - {key}")


    trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in new_model.parameters())

    print(f"\n{'=' * 70}")
    print("Model Rebuild Summary:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Trainable ratio:       {100 * trainable_params / total_params:.4f}%")
    print(f"  Active LoRA modules:   {len(new_submodule_alphas)}")
    print("=" * 70 + "\n")

    return new_model, new_submodule_alphas

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


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



def print_model_parameters(model, submodule_alphas):
    print("\n" + "=" * 80)
    print("Model Parameter Summary".center(80))
    print("=" * 80)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\n[Overall Statistics]")
    print(f"  Total parameters:      {total_params:>15,}  (100.00%)")
    print(f"  Trainable parameters:  {trainable_params:>15,}  ({100 * trainable_params / total_params:>6.2f}%)")
    print(f"  Frozen parameters:     {frozen_params:>15,}  ({100 * frozen_params / total_params:>6.2f}%)")

    print("\n" + "-" * 80)
    print("[Trainable Parameters Breakdown]")

    lora_A_params = 0
    lora_B_params = 0
    alpha_params = 0
    other_params = 0

    param_details = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()

            if 'lora_A' in name:
                lora_A_params += param_count
                param_details.append(('LoRA_A', name, param_count, param.shape))
            elif 'lora_B' in name:
                lora_B_params += param_count
                param_details.append(('LoRA_B', name, param_count, param.shape))
            elif 'alpha' in name:
                alpha_params += param_count
                param_details.append(('Alpha', name, param_count, param.shape))
            else:
                other_params += param_count
                param_details.append(('Other', name, param_count, param.shape))

    total_lora_params = lora_A_params + lora_B_params

    print(f"\n  LoRA parameters:")
    print(f"    - LoRA_A matrices:   {lora_A_params:>15,}  ({100 * lora_A_params / trainable_params:>6.2f}%)")
    print(f"    - LoRA_B matrices:   {lora_B_params:>15,}  ({100 * lora_B_params / trainable_params:>6.2f}%)")
    print(f"    - Total LoRA:        {total_lora_params:>15,}  ({100 * total_lora_params / trainable_params:>6.2f}%)")

    print(f"\n  Architecture parameters:")
    print(f"    - Alpha parameters:  {alpha_params:>15,}  ({100 * alpha_params / trainable_params:>6.2f}%)")
    print(f"    - Number of alphas:  {len(submodule_alphas):>15}")

    if other_params > 0:
        print(f"\n  Other parameters:      {other_params:>15,}  ({100 * other_params / trainable_params:>6.2f}%)")

    print("\n" + "=" * 80 + "\n")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'lora_A_params': lora_A_params,
        'lora_B_params': lora_B_params,
        'alpha_params': alpha_params,
        'other_params': other_params
    }


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_bilevel(lora_model, loader_train_1, loader_train_2, loader_test,
                  w_optim, architect, submodule_alphas,
                  selected_submodules=None):
    losses = []
    loss_log = {'Loss': []}
    loss_log_tmp = {'Loss': []}
    psnr_log = []
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    total_epochs = opt.epochs
    switch_epoch = total_epochs - 25
    switched = False

    steps_per_epoch = len(loader_train_1)

    print(f"\n{'=' * 70}")
    print(f"Bilevel Optimization Training Plan:")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Search phase: Epochs 1-{switch_epoch}")
    print(f"  Fine-tuning phase: Epochs {switch_epoch + 1}-{total_epochs}")
    print("=" * 70 + "\n")

    for epoch in range(1, total_epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Starting Epoch {epoch}/{total_epochs}")
        print("=" * 70)

        if epoch == switch_epoch + 1 and not switched:
            print("\n" + "🔄" * 35)
            print(f"SWITCHING TO FINE-TUNING MODE at Epoch {epoch}")
            print("🔄" * 35)

            if selected_submodules is None:
                submodule_gates = []
                for submodule_name, alpha_param in submodule_alphas.items():
                    gate_value = torch.sigmoid(alpha_param).item()
                    submodule_gates.append((submodule_name, gate_value))

                submodule_gates_sorted = sorted(submodule_gates,
                                                key=lambda x: x[1], reverse=True)
                selected_submodules = [item[0] for item in
                                       submodule_gates_sorted[:opt.top_k_lora]]

            print(f"\nSelected submodules for fine-tuning:")
            for idx, submodule_name in enumerate(selected_submodules, 1):
                print(f"  {idx}. {submodule_name}")

            current_lora_dict, _ = get_top_k_lora_state_dict(
                lora_model,
                submodule_alphas,
                len(selected_submodules)
            )

            base_model = Backbonecs().to(opt.device)
            base_model.load_state_dict(
                torch.load('experiment/DEA/Base/saved_model/best_s.pth')
            )
            for param in base_model.parameters():
                param.requires_grad = False

            lora_model, submodule_alphas = rebuild_model_with_selected_lora(
                base_model=base_model,
                selected_submodules=selected_submodules,
                lora_state_dict=current_lora_dict,
                r=opt.lora_r,
                lora_alpha=opt.lora_alpha
            )
            lora_model = lora_model.to(opt.device)

            trainable_params = [p for p in lora_model.parameters()
                                if p.requires_grad]
            w_optim = optim.Adam(
                params=trainable_params,
                lr=opt.start_lr,
                betas=getattr(opt, 'betas', (0.9, 0.999)),
                eps=getattr(opt, 'eps', 1e-8)
            )

            opt_config = {
                "arch_learning_rate": getattr(opt, 'alpha_lr', 3e-4),
                "arch_weight_decay": getattr(opt, 'alpha_weight_decay', 1e-3),
                "momentum": getattr(opt, 'w_momentum', 0.9),
                "weight_decay": getattr(opt, 'w_weight_decay', 3e-4),
            }
            architect = Architect(lora_model, submodule_alphas, opt_config)

            print_model_parameters(lora_model, submodule_alphas)
            switched = True

        lora_model.train()
        epoch_losses = []

        train_iter_2 = iter(loader_train_2)

        for batch_idx, (trn_X, trn_y) in enumerate(loader_train_1):
            global_step = (epoch - 1) * steps_per_epoch + batch_idx + 1


            lr = opt.start_lr
            if not opt.no_lr_sche:
                total_steps = total_epochs * steps_per_epoch
                lr = lr_schedule_cosdecay(global_step, total_steps)
                for param_group in w_optim.param_groups:
                    param_group["lr"] = lr

            trn_X = trn_X.to(opt.device)
            trn_y = trn_y.to(opt.device)

            try:
                val_X, val_y = next(train_iter_2)
            except StopIteration:
                train_iter_2 = iter(loader_train_2)
                val_X, val_y = next(train_iter_2)

            val_X = val_X.to(opt.device)
            val_y = val_y.to(opt.device)

            if not switched:
                architect.step(
                    input_train=trn_X,
                    target_train=None,
                    input_valid=val_X,
                    target_valid=None,
                    eta=lr,
                    network_optimizer=w_optim,
                    unrolled=True
                )

            w_optim.zero_grad()
            pred = lora_model(trn_X)
            loss = directional_clip_loss(
            output_images=pred,
            input_images=trn_X,
            clip_model=clip_model,
            text_features=text_features
        )
            loss = opt.w_loss_H2C * loss
            loss.backward()

            nn.utils.clip_grad_norm_(
                [p for p in lora_model.parameters() if p.requires_grad],
                getattr(opt, 'w_grad_clip', 5.0)
            )

            w_optim.step()

            losses.append(loss.item())
            loss_log_tmp['Loss'].append(loss.item())
            epoch_losses.append(loss.item())

            mode_str = "Fine-tune" if switched else "Bilevel"
            print(
                f'\r| Mode:{mode_str} | Loss:{loss.item():.5f} | '
                f'Epoch:{epoch}/{total_epochs} | Batch:{batch_idx + 1}/{steps_per_epoch} | '
                f'lr:{lr:.9f} | time:{(time.time() - start_time) / 60:.1f}min',
                end='', flush=True
            )

        for key in loss_log.keys():
            loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
            loss_log_tmp[key] = []
        np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\n[Epoch {epoch}] Average Loss: {avg_epoch_loss:.5f}")

        print(f"\n{'=' * 70}")
        print(f"Evaluating at Epoch {epoch}...")
        print("=" * 70)

        arch_weights = log_architecture_weights(lora_model, submodule_alphas, epoch)

        arch_log_path = os.path.join(opt.saved_data_dir, 'architecture_log.npy')
        if os.path.exists(arch_log_path):
            arch_log = np.load(arch_log_path, allow_pickle=True).item()
        else:
            arch_log = {}
        arch_log[f'epoch_{epoch}'] = arch_weights
        np.save(arch_log_path, arch_log)

        with torch.no_grad():
            ssim_eval, psnr_eval = test(lora_model, loader_test)

        mode_tag = "[Fine-tune]" if switched else "[Bilevel]"
        log = (f'{mode_tag} Epoch:{epoch}/{total_epochs} | '
               f'SSIM:{ssim_eval:.4f} | PSNR:{psnr_eval:.4f}')
        print(log)

        with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
            f.write(log + '\n')

        ssims.append(ssim_eval)
        psnrs.append(psnr_eval)
        psnr_log.append(psnr_eval)

        if switched:
            lora_state_dict = get_lora_state_dict(lora_model)
            top_k_submodules = set(selected_submodules)
        else:
            top_k = opt.top_k_lora
            lora_state_dict, top_k_submodules = get_top_k_lora_state_dict(
                lora_model,
                submodule_alphas,
                top_k
            )

        checkpoint_to_save = {
            'lora_state_dict': lora_state_dict,
            'lora_config': {
                'selected_layers': list(top_k_submodules),
                'r': opt.lora_r,
                'lora_alpha': opt.lora_alpha,
                'top_k': len(top_k_submodules),
                'is_finetuned': switched,
                'epoch': epoch,
            }
        }

        if psnr_eval > max_psnr:
            max_ssim = max(max_ssim, ssim_eval)
            max_psnr = max(max_psnr, psnr_eval)

            print(f'\n{"*" * 70}')
            print(f'✨ New Best Model! Epoch:{epoch}')
            print(f'Max PSNR: {max_psnr:.4f} | Max SSIM: {max_ssim:.4f}')
            print(f'Mode: {"Fine-tuning" if switched else "Bilevel Optimization"}')
            print(f'Active LoRA modules: {len(top_k_submodules)}')
            print("*" * 70 + "\n")

            best_model_path = os.path.join(
                opt.saved_model_dir,
                f'best_lora_adapter_{"finetuned" if switched else "bilevel"}.pth'
            )
            torch.save(checkpoint_to_save, best_model_path)

        epoch_model_path = os.path.join(
            opt.saved_model_dir,
            f'epoch_{epoch}_lora_adapter_{"finetuned" if switched else "bilevel"}.pth'
        )
        torch.save(checkpoint_to_save, epoch_model_path)

        np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
        np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)

        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    from torch.autograd import Variable

    set_seed_torch(3407)

    print("\n" + "=" * 70)
    print("Bilevel Optimization Training")
    print("=" * 70)

    print("\nLoRA Configuration:")
    print(f"  - LoRA Rank (r):      {opt.lora_r}")
    print(f"  - LoRA Alpha:         {opt.lora_alpha}")
    print(f"  - Scaling Factor:     {opt.lora_alpha / opt.lora_r:.4f}")
    print(f"  - Top-K to Save:      {opt.top_k_lora}")
    print(f"  - Total Epochs:       {opt.epochs}")
    print("=" * 70 + "\n")

    start_time = time.time()

    print("Loading CLIP models...")
    clip_model, _ = clip.load(
        "ViT-B/32",
        device=torch.device("cpu"),
        download_root="./clip_model/"
    )
    clip_model.to(opt.device)
    for param in clip_model.parameters():
        param.requires_grad = False

    res_model, _ = clip.load(
        "RN101",
        device=torch.device("cpu"),
        download_root="./clip_model/"
    )
    res_model.to(opt.device)
    for param in res_model.parameters():
        param.requires_grad = False

    text_encoder = TextEncoder(clip_model)
    print("Generating text features from manual prompts...")
    prompts = [
        "a photo with haze",
        "a clear photo"
    ]

    with torch.no_grad():
        tokenized_prompts = clip.tokenize(prompts).to(opt.device)
        text_features = clip_model.encode_text(tokenized_prompts)

    print("Text features generated successfully.")

    clip_model.eval()
    res_model.eval()
    print("✓ CLIP models loaded\n")

    print("Loading backbone model...")
    model = Backbonecs().to(opt.device)
    model.load_state_dict(
        torch.load('experiment/DEA/Base/saved_model/best_s.pth')
    )

    print("Freezing all backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False
    print("✓ Backbone loaded and frozen\n")

    lora_search_space = [
        'down_level1_block1', 'down_level1_block2', 'down_level1_block3', 'down_level1_block4',
        'down_level2_block1', 'down_level2_block2', 'down_level2_block3', 'down_level2_block4',
        'level3_block1', 'level3_block2', 'level3_block3', 'level3_block4',
        'level3_block5', 'level3_block6', 'level3_block7', 'level3_block8'
    ]

    manual_selected_layers = None

    print(f"LoRA search space: {lora_search_space}")
    print(f"Applying LoRA with r={opt.lora_r}, alpha={opt.lora_alpha}\n")

    lora_model, submodule_alphas = apply_lora(
        model=model,
        target_submodules=lora_search_space,
        r=opt.lora_r,
        lora_alpha=opt.lora_alpha
    )
    lora_model = lora_model.to(opt.device)

    param_stats = print_model_parameters(lora_model, submodule_alphas)


    print("Preparing data loaders...")
    train_dir_1 = '/dataset/Real'
    test_dir = '/dataset/Haze4K/val'

    full_set = CLIP_loader(train_dir_1, True, 224)
    total_size = len(full_set)
    train_size = total_size // 2
    val_size = total_size - train_size
    train_set_1, train_set_2 = torch.utils.data.random_split(
        full_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(3407)
    )
    print(f"Dataset split: {train_size} train / {val_size} val (total={total_size})")

    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'), '.png')

    loader_train_1 = DataLoader(
        dataset=train_set_1,
        batch_size=24,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    loader_train_2 = DataLoader(
        dataset=train_set_2,
        batch_size=24,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    loader_test = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    print(f"✓ Train batches per epoch (w update): {len(loader_train_1)}")
    print(f"✓ Train batches per epoch (α update): {len(loader_train_2)}")
    print(f"✓ Test samples: {len(test_set)}\n")

    print("Setting up loss function and optimizers...")


    lora_params = [p for p in lora_model.parameters()
                   if p.requires_grad and 'alpha' not in str(p)]
    w_optimizer = optim.Adam(
        params=lora_params,
        lr=opt.start_lr,
        betas=getattr(opt, 'betas', (0.9, 0.999)),
        eps=getattr(opt, 'eps', 1e-8)
    )

    opt_config = {
        "arch_learning_rate": getattr(opt, 'alpha_lr', 3e-4),
        "arch_weight_decay": getattr(opt, 'alpha_weight_decay', 1e-3),
        "momentum": getattr(opt, 'w_momentum', 0.9),
        "weight_decay": getattr(opt, 'w_weight_decay', 3e-4),
    }
    architect = Architect(lora_model, submodule_alphas, opt_config)

    print(f"✓ Weight Optimizer: Adam (lr={opt.start_lr})")
    print(f"✓ Architecture Optimizer: AdamW (lr={opt_config['arch_learning_rate']})")
    print(f"✓ Architect created for bilevel optimization\n")

    os.makedirs(opt.saved_model_dir, exist_ok=True)
    os.makedirs(opt.saved_data_dir, exist_ok=True)
    print(f"✓ Model save directory: {opt.saved_model_dir}")
    print(f"✓ Data save directory: {opt.saved_data_dir}\n")

    print("=" * 70)
    print("Starting Bilevel Optimization Training")
    print("=" * 70 + "\n")

    w_optimizer.zero_grad()

    train_bilevel(
        lora_model,
        loader_train_1,
        loader_train_2,
        loader_test,
        w_optimizer,
        architect,
        submodule_alphas,
        selected_submodules=manual_selected_layers
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("=" * 70)

