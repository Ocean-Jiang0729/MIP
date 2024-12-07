import os
import sys
import time
sys.path.append(os.path.abspath(__file__ + '/../..'))
from argparse import ArgumentParser

from basicts import launch_runner, BaseRunner


def inference(cfg: dict, runner: BaseRunner, ckpt: str = None, batch_size: int = 1):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')
    # init model
    cfg.TEST.DATA.BATCH_SIZE = batch_size
    runner.model.eval()
    runner.setup_graph(cfg=cfg, train=False)
    # load model checkpoint
    print("@@@", ckpt)
    runner.load_model(ckpt_path=ckpt)
    # inference & speed
    t0 = time.perf_counter()
    runner.test_process(cfg)
    elapsed = time.perf_counter() - t0

    print('##############################')
    runner.logger.info('%s: %0.8fs' % ('Speed', elapsed))
    runner.logger.info('# Param: {0}'.format(sum(p.numel() for p in runner.model.parameters() if p.requires_grad)))

if __name__ == '__main__':
    MODEL_NAME = 'AGCRN'
    DATASET_NAME = 'PEMS08'
    BATCH_SIZE = 32
    GPUS = '2'

    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-m', '--model', default=MODEL_NAME, help='model name')
    parser.add_argument('-d', '--dataset', default=DATASET_NAME, help='dataset name')
    parser.add_argument('-g', '--gpus', default=GPUS, help='visible gpus')
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help='batch size')
    args = parser.parse_args()
    
    cfg_path = 'baselines/{0}/{1}.py'.format(args.model, args.dataset)
    
    ckpt_dir = 'checkpoints/{0}_100/{1}'.format(args.model, args.dataset)
    #print("dir list:",os.listdir(ckpt_dir))
    files = os.listdir(ckpt_dir)
    ckpt_dir = sorted(files)
    #print("ckpt_dir", ckpt_dir)
    ckpt_file = ckpt_dir[0]
    

    #files_with_dates = [(file, os.path.getmtime(os.path.join(ckpt_dir, file))) for file in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, file))]
    # 按修改时间排序，最近修改的文件排在最前
    #files_with_dates.sort(key=lambda x: x[1], reverse=True)
    # 只保留文件名
    #ckpt_file = [file for file, _ in files_with_dates][0]
    # 打印结果
    #print(ckpt_file)

    print("@@@@@", ckpt_file)
    ckpt_path = 'checkpoints/{0}_100/{1}/{2}/{0}_best_val_MAE.pt'.format(args.model, args.dataset, ckpt_file )
    print("#####", ckpt_path)
    launch_runner(cfg_path, inference, (ckpt_path, args.batch_size), devices=args.gpus)
