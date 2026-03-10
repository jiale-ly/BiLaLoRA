import os
import random
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, Resize, CenterCrop
from torchvision.transforms import functional as FF


def preprocess_feature(img):
    img = ToTensor()(img)
    clip_normalizer = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img = clip_normalizer(img)
    return img


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=224, format='.jpg'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1].split('_')
        id = split_name[0]
        if self.format:
            clear_name = id + self.format
        else:
            clear_name = id
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.haze_imgs)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, format='.jpg'):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()
        self.size = 224
        self.format = format

    def __getitem__(self, index):

        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0]

        if self.format:
            clear_image_name += self.format

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        if hazy.size[0] < self.size or hazy.size[1] < self.size:
            hazy = Resize((self.size, self.size))(hazy)
            clear = Resize((self.size, self.size))(clear)

        i, j, h, w = RandomCrop.get_params(hazy, output_size=(self.size, self.size))
        hazy = FF.crop(hazy, i, j, h, w)
        clear = FF.crop(clear, i, j, h, w)

        hazy, clear = self.augData(hazy.convert("RGB"), clear.convert("RGB"))

        return hazy, clear, hazy_image_name

    def augData(self, data, target):
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.hazy_image_list)

class TestDatasetSX(data.Dataset):
    def __init__(self, hazy_path, clear_path, format='.jpg'):
        super(TestDatasetSX, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()
        self.size = 256
        self.format = format

    def __getitem__(self, index):

        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0]



        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        if hazy.size[0] < self.size or hazy.size[1] < self.size:
            hazy = Resize((self.size, self.size))(hazy)
            clear = Resize((self.size, self.size))(clear)

        i, j, h, w = RandomCrop.get_params(hazy, output_size=(self.size, self.size))
        hazy = FF.crop(hazy, i, j, h, w)
        clear = FF.crop(clear, i, j, h, w)

        hazy, clear = self.augData(hazy.convert("RGB"), clear.convert("RGB"))

        return hazy, clear, hazy_image_name

    def augData(self, data, target):
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.hazy_image_list)


class CLIP_loader(data.Dataset):

    def __init__(self, hazy_path, train, size=256):
        self.hazy_path = hazy_path
        self.train = train
        self.hazy_image_list = os.listdir(hazy_path)
        self.hazy_image_list.sort()
        self.size = size

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        width, height = hazy.size
        crop_size = min(self.size, height, width)

        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(hazy, output_size=(crop_size, crop_size))
            hazy = FF.crop(hazy, i, j, h, w)
        hazy = Resize((self.size, self.size))(hazy)
        haze, haze_r = self.augData(hazy.convert("RGB"))
        return haze, haze_r

    def augData(self, data):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
        return preprocess_feature(data), ToTensor()(data)

    def __len__(self):
        return len(self.hazy_image_list)

class CLIP_loader2(data.Dataset):

    def __init__(self, hazy_path, train):
        self.hazy_path = hazy_path
        self.train = train
        self.hazy_image_list = os.listdir(hazy_path)
        self.hazy_image_list.sort()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        haze, haze_r = self.augData(hazy.convert("RGB"))
        return haze, haze_r

    def augData(self, data):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
        return preprocess_feature(data), ToTensor()(data)

    def __len__(self):
        return len(self.hazy_image_list)

class RESIDE_Dataset_2(data.Dataset):
    def __init__(self, path, train, size=256, format='.jpg'):
        super(RESIDE_Dataset_2, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 100)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        split_name = os.path.split(img)[-1]  # 获取文件名和后缀部分

        # 提取雾图的文件基础名（包含下划线，但去掉后缀）
        id = os.path.splitext(split_name)[0]  # 去掉后缀名，保留文件基础名

        # 构造清晰图的文件名
        clear_name = f"{id}{self.format}"  # 假设 self.format 是清晰图的文件后缀名，例如 '.png'
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        if not isinstance(self.size, str):
            i, j, h, w = RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.haze_imgs)

class TestDataset_OTS(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset_OTS, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()
        self.size = 256

    def __getitem__(self, index):
        # data shape: C*H*W
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        if hazy.size[0] < self.size or hazy.size[1] < self.size:
            hazy = Resize((self.size, self.size))(hazy)
            clear = Resize((self.size, self.size))(clear)

        # Proceed with random crop
        i, j, h, w = RandomCrop.get_params(hazy, output_size=(self.size, self.size))
        hazy = FF.crop(hazy, i, j, h, w)
        clear = FF.crop(clear, i, j, h, w)

        hazy, clear = self.augData(hazy.convert("RGB"), clear.convert("RGB"))

        return hazy, clear, hazy_image_name

    def augData(self, data, target):
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.hazy_image_list)


class RESIDE_Dataset_DH(data.Dataset):
    def __init__(self, path, train, size=256, format='.jpg', degraded_types=None):
        super(RESIDE_Dataset_DH, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.base_path = path

        # 所有可用的退化类型
        all_degraded_types = [
            "color", "color2dark", "color2dark2noise", "color2noise",
            "dark2color", "dark2color2noise"
        ]

        # 确定要使用的退化类型
        if degraded_types is None:
            self.degraded_types = all_degraded_types
        else:
            # 验证输入的退化类型是否有效
            valid_types = [t for t in degraded_types if t in all_degraded_types]
            if len(valid_types) < len(degraded_types):
                invalid_types = set(degraded_types) - set(valid_types)
                print(f"警告: 忽略无效的退化类型: {invalid_types}")
            self.degraded_types = valid_types

        # 收集所有退化图像路径
        self.degraded_imgs = []
        for folder in self.degraded_types:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                for img in os.listdir(folder_path):
                    if img.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.degraded_imgs.append(os.path.join(folder_path, img))
            else:
                print(f"警告: 退化文件夹不存在 - {folder_path}")

        # 清晰图像目录
        self.clear_dir = os.path.join(self.base_path, 'clear')

        print(f"数据集初始化完成: 使用 {len(self.degraded_types)} 种退化类型")
        print(f"  找到 {len(self.degraded_imgs)} 张退化图像")
        print(f"  退化类型: {', '.join(self.degraded_types)}")

    def __getitem__(self, index):
        # 加载退化图像
        degraded_path = self.degraded_imgs[index]
        degraded = Image.open(degraded_path)

        # 确保图像尺寸足够
        if isinstance(self.size, int):
            while degraded.size[0] < self.size or degraded.size[1] < self.size:
                index = random.randint(0, len(self.degraded_imgs) - 1)
                degraded_path = self.degraded_imgs[index]
                degraded = Image.open(degraded_path)

        # 提取图像ID（文件名格式为：ID_0_退化类型.扩展名）
        filename = os.path.basename(degraded_path)
        id_part = filename.split('_')[0]  # 提取ID部分

        # 构建清晰图像路径
        clear_path = os.path.join(self.clear_dir, f"{id_part}{self.format}")
        if not os.path.exists(clear_path):
            # 尝试其他可能的格式
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                clear_path = os.path.join(self.clear_dir, f"{id_part}{ext}")
                if os.path.exists(clear_path):
                    break

        # 加载清晰图像
        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"找不到对应的清晰图像: {id_part}{self.format}")

        clear = Image.open(clear_path)

        # 随机裁剪
        if not isinstance(self.size, str):
            # 确保图像足够大
            if degraded.size[0] < self.size or degraded.size[1] < self.size:
                # 如果图像太小，调整尺寸
                degraded = FF.resize(degraded, (self.size, self.size))
                clear = FF.resize(clear, (self.size, self.size))
            else:
                # 随机裁剪
                i, j, h, w = RandomCrop.get_params(degraded, output_size=(self.size, self.size))
                degraded = FF.crop(degraded, i, j, h, w)
                clear = FF.crop(clear, i, j, h, w)

        # 数据增强
        degraded, clear = self.augData(degraded.convert("RGB"), clear.convert("RGB"))
        return degraded, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = RandomHorizontalFlip(rand_hor)(data)
            target = RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        target = ToTensor()(target)
        return preprocess_feature(data), target

    def __len__(self):
        return len(self.degraded_imgs)


class TestDatasetDH(data.Dataset):
    def __init__(self, path, format='.jpg', degraded_types=None, size=256):
        super(TestDatasetDH, self).__init__()
        self.base_path = path
        self.size = size
        self.format = format

        # 所有可用的退化类型
        all_degraded_types = [
            "color", "color2dark", "color2dark2noise", "color2noise", "dark",
            "dark2color", "dark2color2noise", "dark2haze", "dark2haze2noise",
            "dark2noise", "haze", "haze2dark", "haze2dark2noise",
            "haze2noise", "noise"
        ]

        # 确定要使用的退化类型
        if degraded_types is None:
            self.degraded_types = all_degraded_types
        else:
            # 验证输入的退化类型是否有效
            valid_types = [t for t in degraded_types if t in all_degraded_types]
            if len(valid_types) < len(degraded_types):
                invalid_types = set(degraded_types) - set(valid_types)
                print(f"警告: 忽略无效的退化类型: {invalid_types}")
            self.degraded_types = valid_types

        # 收集所有退化图像路径
        self.degraded_imgs = []
        for folder in self.degraded_types:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                for img in os.listdir(folder_path):
                    if img.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.degraded_imgs.append(os.path.join(folder_path, img))
            else:
                print(f"警告: 退化文件夹不存在 - {folder_path}")

        # 清晰图像目录
        self.clear_dir = os.path.join(self.base_path, 'clear')

        # 排序确保顺序一致
        self.degraded_imgs.sort()

        print(f"测试集初始化完成: 使用 {len(self.degraded_types)} 种退化类型")
        print(f"  找到 {len(self.degraded_imgs)} 张退化图像")
        print(f"  退化类型: {', '.join(self.degraded_types)}")

        # 预处理变换
        self.resize = Resize(size) if size else None
        self.center_crop = CenterCrop(size) if size else None

    def __getitem__(self, index):
        # 加载退化图像
        degraded_path = self.degraded_imgs[index]
        degraded = Image.open(degraded_path).convert('RGB')

        # 提取图像ID（文件名格式为：ID_0_退化类型.扩展名）
        filename = os.path.basename(degraded_path)
        id_part = filename.split('_')[0]  # 提取ID部分

        # 构建清晰图像路径
        clear_path = os.path.join(self.clear_dir, f"{id_part}{self.format}")
        if not os.path.exists(clear_path):
            # 尝试其他可能的格式
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                clear_path = os.path.join(self.clear_dir, f"{id_part}{ext}")
                if os.path.exists(clear_path):
                    break

        # 加载清晰图像
        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"找不到对应的清晰图像: {id_part}{self.format}")

        clear = Image.open(clear_path).convert('RGB')

        # 调整图像尺寸
        if self.size:
            # 如果图像尺寸小于目标尺寸，则调整大小
            if degraded.width < self.size or degraded.height < self.size:
                degraded = self.resize(degraded)
                clear = self.resize(clear)
            # 中心裁剪到目标尺寸
            degraded = self.center_crop(degraded)
            clear = self.center_crop(clear)

        # 转换为张量
        degraded_tensor = preprocess_feature(degraded)
        clear_tensor = ToTensor()(clear)

        # 返回退化图像、清晰图像和原始文件名
        return degraded_tensor, clear_tensor, filename

    def __len__(self):
        return len(self.degraded_imgs)