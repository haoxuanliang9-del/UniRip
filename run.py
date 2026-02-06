import torch
import logging
from mmkgc.config import Tester, Trainer
from mmkgc.module.model import RotatE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader

from args import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)

    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
                  
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
                             
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
                         
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')
    
                                                          
    import os
    num_path = './embeddings/' + args.dataset + '-numeric.pth'
    num_emb = torch.load(num_path) if os.path.exists(num_path) else None

    def build_adj_tensor(triple_path, ent_tot, max_neighbors):
        adj_entities = [[] for _ in range(ent_tot)]
        adj_relations = [[] for _ in range(ent_tot)]
        with open(triple_path, "r") as f:
            total = int(f.readline().strip())
            for line in f:
                h, t, r = map(int, line.strip().split())
                adj_entities[h].append(t)
                adj_relations[h].append(r)
                adj_entities[t].append(h)
                adj_relations[t].append(r)

        adj_ent_tensor = torch.zeros(ent_tot, max_neighbors, dtype=torch.long)
        adj_rel_tensor = torch.zeros(ent_tot, max_neighbors, dtype=torch.long)
        neighbor_mask = torch.zeros(ent_tot, max_neighbors, dtype=torch.bool)

        for eid, (neis, rels) in enumerate(zip(adj_entities, adj_relations)):
            if len(neis) == 0:
                continue
            capped = neis[:max_neighbors]
            rel_capped = rels[:max_neighbors]
            n = len(capped)
            adj_ent_tensor[eid, :n] = torch.tensor(capped, dtype=torch.long)
            adj_rel_tensor[eid, :n] = torch.tensor(rel_capped, dtype=torch.long)
            neighbor_mask[eid, :n] = True

        return adj_ent_tensor, adj_rel_tensor, neighbor_mask

    adj_entities, adj_relations, neighbor_mask = build_adj_tensor(
        "./benchmarks/" + args.dataset + "/train2id.txt",
        train_dataloader.get_ent_tot(),
        args.max_neighbor
    )

                      
    kge_score = RotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb,
        num_emb=num_emb,
        adj_entities=adj_entities,
        adj_relations=adj_relations,
        neighbor_mask=neighbor_mask,
        max_neighbors=args.max_neighbor
    )
    print(kge_score)
                              
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=0.00001,
        struct_cons_rate=args.lamda,
        modal_align_rate=args.modal_align_rate
    )
    
                     
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)

    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        weight_decay=args.weight_decay,
        mu=args.mu,
        tester=tester,
        test_interval=50,
        early_stop_delta=0.01,
        early_stop_patience=args.loss_early_stop_patience,
        metric_name=args.metric,
        metric_delta=args.metric_delta,
        metric_patience=args.metric_patience,
        lr_decay_patience=args.lr_decay_patience,
        lr_decay_factor=args.lr_decay_factor,
        checkpoint_dir=args.save
    )

    trainer.run()
    kge_score.save_checkpoint(args.save)

                    
    kge_score.load_checkpoint(args.save)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
