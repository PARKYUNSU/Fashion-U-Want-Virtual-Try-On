import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw

import os.path as osp
import numpy as np
import json


class CPDatasetTest(data.Dataset):
    """
        Test Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDatasetTest, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names['paired'] = im_names
        self.c_names['unpaired'] = c_names

    def name(self):
        return "CPDataset"

    def get_parse_agnostic(self, parse, pose_data):
        """
        Generate agnostic parse map by masking specific body parts.
        """
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        """
        Load the image, cloth, mask, parse, and other related data.
        """
        im_name = self.im_names[index]
        c_name = {}
        c = {}
        cm = {}

        # Load cloth and mask
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            if key == "paired":
                c[key] = Image.open(osp.join(self.data_path, 'image', c_name[key])).convert('RGB')
            else:
                c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')
            c[key] = transforms.Resize((self.fine_height, self.fine_width), interpolation=2)(c[key])
            if key == "paired":
                cm[key] = Image.open(osp.join(self.data_path, 'image-parse-v3', c_name[key]).replace('.jpg', '.png'))
            else:
                cm[key] = Image.open(osp.join(self.data_path, 'cloth-mask', c_name[key]))
            cm[key] = transforms.Resize((self.fine_height, self.fine_width), interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # Normalize to [-1, 1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array).unsqueeze_(0)  # [0, 1]

        # Load person image
        im_pil_big = Image.open(osp.join(self.data_path, 'image', im_name))
        im_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # Load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse_pil_big = Image.open(osp.join(self.data_path, 'image-parse-v3', parse_name))
        im_parse_pil = transforms.Resize((self.fine_height, self.fine_width), interpolation=0)(im_parse_pil_big)

        # Generate agnostic map
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = np.array(pose_label['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:, :2]

        parse_agnostic = self.get_parse_agnostic(im_parse_pil, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()

        # Prepare densepose data (if used)
        densepose_name = im_name.replace('image', 'image-densepose')
        densepose_map = Image.open(osp.join(self.data_path, 'image-densepose', densepose_name))
        densepose_map = transforms.Resize((self.fine_height, self.fine_width), interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)

        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth
            'cloth': c,  # for input
            'cloth_mask': cm,  # for input
            'parse_agnostic': parse_agnostic,  # agnostic map
            'densepose': densepose_map,  # densepose map
            'image': im,  # original person image
        }

        return result

    def __len__(self):
        return len(self.im_names)


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()
        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
