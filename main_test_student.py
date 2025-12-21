import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

# Import YOUR Student Architecture
from models.network_swinir_student import SwinIR_Student as net
from utils import util_calculate_psnr_ssim as util


def main():
    parser = argparse.ArgumentParser()
    # Task: Default to classical_sr since that is your project
    parser.add_argument('--task', type=str, default='classical_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    # Scale: Default to 4 (Your project is x4)
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    
    # IMPORTANT: Removed default path. You MUST provide the path to your trained brain.
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained student model (.pth). REQUIRED.')
    
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # 1. Setup Model
    # ----------------------------------------
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model not found at {args.model_path}. Please check the filename.')
    
    print(f'Loading model from {args.model_path}')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # ----------------------------------------
    # 2. Setup Folders
    # ----------------------------------------
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

    # ----------------------------------------
    # 3. Inference Loop
    # ----------------------------------------
    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            
            # Run the Student Model
            output = test(img_lq, model, args, window_size)
            
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            
            # Print results for this image
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # ----------------------------------------
    # 4. Summary
    # ----------------------------------------
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


def define_model(args):
    # Setup for our Lightweight Student (Model C / A / B)
    if args.task in ['classical_sr', 'lightweight_sr']:
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[4, 4, 4, 4], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    else:
        # Fallback for other tasks if you ever use them
        # (This keeps the code from breaking if you accidentally pick 'gray_dn')
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[4, 4, 4, 4], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')

    # Load the Trained Brain
    pretrained_model = torch.load(args.model_path)
    param_key = 'params' 
    if param_key in pretrained_model:
        model.load_state_dict(pretrained_model[param_key], strict=True)
    else:
        model.load_state_dict(pretrained_model, strict=True)

    return model


def setup(args):
    # Get the model filename (e.g., "165000_E") without the folder path or .pth extension
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Define where the results get saved
    if args.task in ['classical_sr', 'lightweight_sr']:
        # NOW: results/swinir_classical_sr_x4_165000_E
        save_dir = f'results/swinir_{args.task}_x{args.scale}_{model_name}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8
    else:
        # Fallback
        save_dir = f'results/swinir_{args.task}_{model_name}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # Classical SR: We usually load High-Res (GT) and generate Low-Res on the fly?
    # Actually, in standard testing, we often read pre-generated LR images.
    # The code below assumes you have a folder structure where LR images exist.
    
    # Load GT
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    
    # Try to load LR from a specific folder if provided
    if args.folder_lq:
        # Construct LR filename. Standard is "Namex4.png"
        img_lq_path = f'{args.folder_lq}/{imgname}x{args.scale}{imgext}'
        img_lq = cv2.imread(img_lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    else:
        # If no LR folder provided, we must degrade the GT on the fly (Bicubic)
        # This is strictly for quick testing if you don't have LR dataset ready.
        h, w = img_gt.shape[:2]
        img_lq = cv2.resize(img_gt, (w // args.scale, h // args.scale), interpolation=cv2.INTER_CUBIC)

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole (Standard for Set5)
        output = model(img_lq)
    else:
        # test the image tile by tile (For huge images only)
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()