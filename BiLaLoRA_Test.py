import glob
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode
import math
import os
import torch
import torch.nn.functional as F
from torch import  nn
import torch.utils.data
from DEAmodel import Backbonecs

GENERATE_BASE_MODEL = True
GENERATE_LORA_MODEL = True

BASE_MODEL_PATH = 'weight/Base.pth'
LORA_CHECKPOINT_PATH = 'weight/LoRA.pth'

INPUT_FOLDER = './test_data/500_foggy'
OUTPUT_FOLDER_BASE = './output/500_foggy/base'
OUTPUT_FOLDER_LORA = './output/500_foggy/lora'
# OUTPUT_FOLDER_BASE = './outputs/BiLaLoRA/base/test'
# OUTPUT_FOLDER_LORA = './outputs/BiLaLoRA/lora/test'


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


def inject_lora_to_layers(model, target_layers, r, lora_alpha, device):
    submodule_alphas = {}

    for layer_name in target_layers:
        try:
            parent_module = model.get_submodule(layer_name)
            shared_alpha = nn.Parameter(torch.zeros(1))
            submodule_alphas[layer_name] = shared_alpha

            for child_name, child_module in list(parent_module.named_children()):
                if isinstance(child_module, nn.Conv2d):
                    new_layer = LoRALayer(
                        original_layer=child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        shared_alpha=shared_alpha
                    ).to(device)

                    new_layer.original_layer.weight.data.copy_(child_module.weight.data)
                    if child_module.bias is not None:
                        new_layer.original_layer.bias.data.copy_(child_module.bias.data)

                    setattr(parent_module, child_name, new_layer)
        except Exception as e:
            print(f"Warning: Could not inject LoRA to {layer_name}: {e}")

    return model, submodule_alphas


def dehaze(model, image_path, folder, device, transform):
    haze = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    h, w = haze.shape[2], haze.shape[3]
    haze = Resize((h // 16 * 16, w // 16 * 16), interpolation=InterpolationMode.BICUBIC, antialias=True)(haze)

    with torch.no_grad():
        out = model(haze)

    if isinstance(out, tuple):
        out = out[0]

    out = out.squeeze(0)
    out = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(out)
    torchvision.utils.save_image(out, os.path.join(folder, os.path.basename(image_path)))


def check_output_exists(output_folder, image_list):
    if not os.path.exists(output_folder):
        return False

    existing_files = set(os.listdir(output_folder))
    expected_files = {os.path.basename(img) for img in image_list}

    return expected_files.issubset(existing_files)


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Generate Base Model: {GENERATE_BASE_MODEL}")
    print(f"Generate LoRA Model: {GENERATE_LORA_MODEL}\n")

    images = glob.glob(os.path.join(INPUT_FOLDER, '*jpg')) + \
             glob.glob(os.path.join(INPUT_FOLDER, '*png')) + \
             glob.glob(os.path.join(INPUT_FOLDER, '*jpeg'))

    if GENERATE_BASE_MODEL:
        print("--- (1/2) Base Model Inference ---")

        if check_output_exists(OUTPUT_FOLDER_BASE, images):
            print(f"✓ Base model outputs already exist in: {OUTPUT_FOLDER_BASE}")
            print("⊳ Skipping base model inference\n")
        else:
            base_model = Backbonecs().to(device)
            base_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
            base_model.eval()

            os.makedirs(OUTPUT_FOLDER_BASE, exist_ok=True)
            with torch.no_grad():
                for image in tqdm(images, desc="Base model"):
                    dehaze(base_model, image, OUTPUT_FOLDER_BASE, device, transform)
            print(f"✓ Saved to: {OUTPUT_FOLDER_BASE}\n")
    else:
        print("--- Base Model Inference ---")
        print("⊳ Skipped (GENERATE_BASE_MODEL = False)\n")

    if GENERATE_LORA_MODEL:
        print("--- (2/2) LoRA Model Inference ---")

        print(f"Loading checkpoint from: {LORA_CHECKPOINT_PATH}")
        checkpoint = torch.load(LORA_CHECKPOINT_PATH, map_location=device)

        config = checkpoint['nas_lora_config']
        selected_layers = config['selected_layers']
        lora_r = config['r']
        lora_alpha = config['lora_alpha']
        lora_state_dict = checkpoint['lora_state_dict']

        print(f"Selected layers: {selected_layers}")
        print(f"LoRA config: r={lora_r}, alpha={lora_alpha}\n")

        lora_model = Backbonecs().to(device)
        lora_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))

        print("Injecting LoRA layers...")
        lora_model, _ = inject_lora_to_layers(
            model=lora_model,
            target_layers=selected_layers,
            r=lora_r,
            lora_alpha=lora_alpha,
            device=device
        )
        print(f"✓ LoRA layers injected\n")

        print("Loading LoRA weights...")
        lora_model.load_state_dict(lora_state_dict, strict=False)
        lora_model.eval()

        print(f"✓ Loaded {len(lora_state_dict)} LoRA parameters\n")

        os.makedirs(OUTPUT_FOLDER_LORA, exist_ok=True)
        with torch.no_grad():
            for image in tqdm(images, desc="LoRA model"):
                dehaze(lora_model, image, OUTPUT_FOLDER_LORA, device, transform)
        print(f"✓ Saved to: {OUTPUT_FOLDER_LORA}")
    else:
        print("--- LoRA Model Inference ---")
        print("⊳ Skipped (GENERATE_LORA_MODEL = False)")

    print("\n" + "=" * 70)
    print("Inference completed!")
    print("=" * 70)