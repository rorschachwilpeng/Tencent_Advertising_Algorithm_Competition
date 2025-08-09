import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset as ds  # 用于获取多模态缓存统计
from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))#这个是tensorboard的writer，用于记录训练过程中的指标
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum#用户数量和物品数量
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types#特征统计和特征类型

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)#初始化模型

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)#初始化参数
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0#位置编码
    model.item_emb.weight.data[0, :] = 0#物品编码
    model.user_emb.weight.data[0, :] = 0#用户编码

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0

    # IO/Compute 统计
    log_interval = 200
    data_time_sum = 0.0
    compute_time_sum = 0.0
    t_data_wait_start = time.time()

    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 统计数据获取时间（上一step结束到本step拿到batch的时间）
            data_time = time.time() - t_data_wait_start
            data_time_sum += data_time

            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)

            # 前向与反向
            compute_start = time.time()
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()
            compute_time_sum += time.time() - compute_start

            # 下一次数据等待起点
            t_data_wait_start = time.time()

            # 每隔若干步打印与记录缓存命中率与IO时间
            if global_step % log_interval == 0:
                # 统计缓存
                try:
                    mm_stats = ds._get_mm_cache_stats()#多模态缓存统计
                except Exception:
                    mm_stats = {'hit_rate': 0.0, 'entries': 0}
                item_stats = dataset.get_cache_stats()#物品特征缓存统计
                user_stats = dataset.get_user_cache_stats()#用户特征缓存统计

                avg_data_time = data_time_sum / log_interval
                avg_compute_time = compute_time_sum / log_interval

                print(f"[io] step={global_step} avg_data_time={avg_data_time:.4f}s avg_compute_time={avg_compute_time:.4f}s")
                print(f"[cache] mm_hit={mm_stats.get('hit_rate',0):.2%} entries={mm_stats.get('entries',0)} | "
                      f"item_hit={item_stats['hit_rate']:.2%} entries={item_stats['item_feat_cache_size']} | "
                      f"user_hit={user_stats['hit_rate']:.2%} entries={user_stats['entries']}")

                # TensorBoard
                writer.add_scalar('io/avg_data_time', avg_data_time, global_step)
                writer.add_scalar('io/avg_compute_time', avg_compute_time, global_step)
                writer.add_scalar('cache/mm_hit_rate', mm_stats.get('hit_rate',0), global_step)
                writer.add_scalar('cache/mm_entries', mm_stats.get('entries',0), global_step)
                writer.add_scalar('cache/item_hit_rate', item_stats['hit_rate'], global_step)
                writer.add_scalar('cache/item_entries', item_stats['item_feat_cache_size'], global_step)
                writer.add_scalar('cache/user_hit_rate', user_stats['hit_rate'], global_step)
                writer.add_scalar('cache/user_entries', user_stats['entries'], global_step)

                # reset window sums
                data_time_sum = 0.0
                compute_time_sum = 0.0

        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
