import torch
from torch.utils.data import DataLoader
from datasets import get_cifar10_data, extract_cifar10_images, Cifar10Dataset
from networks import Generator
from helpers import save_test_sample, print_args
import warnings
warnings.simplefilter("ignore") # sorry. warnings annoye me
import argparse
import os
import os.path as osp


def main(args):    
    # print args
    print_args(args)    
    
    # download and extract dataset
    get_cifar10_data(args.data_path)
    data_dirs = extract_cifar10_images(args.data_path)
    
    dataset = Cifar10Dataset(root_dir=data_dirs["test"], mirror=False, random_seed=1) 
    
    print("test dataset len: {}".format(len(dataset)))
      
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    generator_bn = Generator("batch")
    generator_sn = Generator("batch")

    generator_bn.load_state_dict(torch.load("checkpoints/checkpoint_ep0_gen.pt", map_location="cpu"))
    generator_sn.load_state_dict(torch.load("checkpoints_spectral/checkpoint_ep10_gen.pt", map_location="cpu"))
        
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for idx, sample in enumerate(data_loader):
        
        img_l, real_img_lab = sample[:,0:1,:,:], sample
        
        fake_img_ab_bn = generator_bn(img_l).detach()
        fake_img_lab_bn = torch.cat([img_l, fake_img_ab_bn], dim=1)
        
        fake_img_ab_sn = generator_sn(img_l).detach()
        fake_img_lab_sn = torch.cat([img_l, fake_img_ab_sn], dim=1)
        
        print("sample {}/{}".format(idx+1, len(data_loader)+1))
        save_test_sample(real_img_lab, fake_img_lab_bn, fake_img_lab_sn, 
                         osp.join(args.save_path, "test_sample_{}.png".format(idx)), show=True)          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Image colorization with GANs")
    parser.add_argument("--data_path", type=str, default="./data", help="Download and extraction path for the dataset")
    parser.add_argument("--save_path", type=str, default="./output_imgs", help="Save path for the test imgs")   
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    main(args)
