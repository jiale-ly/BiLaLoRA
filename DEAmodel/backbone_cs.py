import torch.nn as nn
from .modules import *
from .LoRA.loralib import layers as lora

def replace_layers_for_deanet(model, desired_submodules, r=2, lora_alpha=2):
    for name, sub_module in model.named_children():
        if name in desired_submodules:
            print(f"-> Targeting submodule: '{name}' of type {type(sub_module).__name__}")
            for child_name, layer in list(sub_module.named_children()):
                if isinstance(layer, nn.Conv2d):
                    print(f"  ... Found Conv2d layer: '{child_name}', replacing with LoRA version.")
                    new_lora_layer = lora.Conv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size[0],
                        stride=layer.stride[0],
                        padding=layer.padding[0],
                        dilation=layer.dilation[0],
                        groups=layer.groups,
                        bias=layer.bias is not None,
                        r=r,
                        lora_alpha=lora_alpha
                    ).to(layer.weight.device)


                    new_lora_layer.weight.data.copy_(layer.weight.data)
                    if layer.bias is not None:
                        new_lora_layer.bias.data.copy_(layer.bias.data)


                    setattr(sub_module, child_name, new_lora_layer)

    return model

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Backbonecs(nn.Module):
    def __init__(self, base_dim=32):
        super(Backbonecs, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlock(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block3 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block6 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlock(default_conv, base_dim * 4, 3)

        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)


    def forward(self, x):
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out


if __name__ == "__main__":

    model = Backbonecs()

    lora_target_submodules = [
        'level3_block1',
        'mix1'
    ]

    print("--- Freezing all model parameters first... ---")
    for param in model.parameters():
        param.requires_grad = False

    print("\n--- Applying LoRA... ---")
    lora_model = replace_layers_for_deanet(
        model=model,
        desired_submodules=lora_target_submodules,
        r=8,
        lora_alpha=16
    )
    print("--- LoRA application finished. ---\n")


    print("\n--- Trainable Parameters (Corrected) ---")
    total_params = 0
    trainable_params = 0
    for name, param in lora_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable: {name}, Size: {param.numel()}")
            print(name, param.device)

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable LoRA parameters: {trainable_params}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.4f}%")