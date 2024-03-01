# # TODO: make a good histogram of the AUCs for each property.
# TODO:Show how AUC change over time. Median AUC for each position with line chart./Another way, for each property, it improves over time.
# TODO:what's the difference between 0 to 1, 1 to 2, 2 to 3, take the median difference.
# TODO: New training for adding inp

import itertools, pathlib, numpy as np, tqdm
import torch, torch.utils.data, torch.optim as optim
import cvae.tokenizer, cvae.utils as utils
import cvae.models.multitask_transformer as mt 

DEVICE = torch.device(f'cuda:0')

# load data ===========================================================================
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('/home/yifan/git/ai.biobricks/cvae/data/processed/selfies_property_val_tokenizer')

class Trainer():
    
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(model.parameters(),lr=1e-3)
        self.lossfn = mt.MultitaskTransfeormer.lossfn()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.9, verbose=True, min_lr=1e-6)
        self.scheduler_loss = []
        self.scheduler_loss_interval = 100
        
        self.savepath = "brick/mtransform2"
        self.test_losses = [np.inf]
        self.best_test_loss = np.inf
    
    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        return self
    
    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        return self
    
    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        return self
    
    def set_metrics_file(self, metrics_path, overwrite=False):
        self.metrics_path = metrics_path
        if overwrite: utils.write_path(self.metrics_path,"epoch\tloss\tlearning_rate\n", mode='w')
        return self
    
    def set_values_mask_flag(self, bool_flag):
        self.values_mask = bool_flag
        return self
    
    def _evaluation_loss(self, valdl):
        self.model.eval()
        epochloss = []
        for _, (inp, teach, out) in tqdm.tqdm(enumerate(valdl), total=len(valdl)):
            inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
            pred = self.model(inp, teach)
            loss = self.lossfn(self.model.parameters(), pred.permute(0,2,1), out)
            epochloss.append(loss.item())
        
        return np.mean(epochloss)
                
    def _epochs_since_improvement(self):
        return len(self.test_losses) - np.argmin(self.test_losses)
    
    def _train_batch(self, inp, teach, out):
        _ = self.model.train()
        self.optimizer.zero_grad()
        
        # outputs and loss
        pred = self.model(inp, teach)
        loss = self.lossfn(self.model.parameters(), pred.permute(0,2,1), out)
        
        # update model
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def start(self):
        i, (inp, teach, out) = next(self.trn_iterator)
        batch_size = inp.size(0)
        
        # evaluate twice per epoch
        evaluation_interval = ((len(trnds)-1) // batch_size) // 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        value_indexes = torch.LongTensor(list(tokenizer.value_indexes().values())).to(DEVICE)
        
        epoch = 0
        trn_loss = []
        
        while self._epochs_since_improvement() < 4000:
            
            i, (inp, teach, out) = next(self.trn_iterator)
            inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
            this_batch_size = inp.size(0)
            
            mask = torch.rand(inp.shape, device=DEVICE) < self.mask_percent
            mask[:,0] = False # don't mask the first token
            inp = inp.masked_fill(mask, tokenizer.pad_idx)
            
            mask = torch.rand(teach.shape, device=DEVICE) < self.mask_percent
            mask[:,0] = False # don't mask the first token
            teach = teach.masked_fill(mask, tokenizer.pad_idx)
            
            if self.values_mask:
                teach = teach.masked_fill(torch.isin(teach, value_indexes), tokenizer.pad_idx)

            loss = self._train_batch(inp, teach, out)
            trn_loss.append(loss)
            utils.write_path(self.metrics_path,f"train\t{i}\t{loss/this_batch_size}\t{self.optimizer.param_groups[0]['lr']:.4f}\n")
            
            # SCHEDULER UPDATE 
            if (i + 1) % self.scheduler_loss_interval == 0:
                mean_loss = np.mean(trn_loss)
                utils.write_path(self.metrics_path,f"scheduler\t{i}\t{mean_loss/this_batch_size}\t{self.optimizer.param_groups[0]['lr']:.4f}\n")
                trn_loss = []
            
            # EVALUATION UPDATE
            if (i + 1) % evaluation_interval == 0:
                epoch += 1
                eval_loss = self._evaluation_loss(valdl)
                self.scheduler.step(mean_loss)
                self.test_losses.append(eval_loss)
                if eval_loss < self.best_test_loss:
                    self.best_test_loss = eval_loss
                    self.model.module.save(self.savepath)
                
                utils.write_path(self.metrics_path,f"eval\t{i}\t{self.test_losses[-1]/batch_size}\n")
                print(f"epoch: {epoch}\t eval_loss: {self.best_test_loss/batch_size:.4f}")
        
model = mt.MultitaskDecoderTransformer(tokenizer).to(DEVICE)
model = torch.nn.DataParallel(model)

trnds = mt.SequenceShiftDataset("data/processed/multitask_tensors/trn", tokenizer.pad_idx, tokenizer.SEP_IDX, tokenizer.END_IDX)
trndl = torch.utils.data.DataLoader(trnds, batch_size=248, shuffle=True, prefetch_factor=100, num_workers=20)
valds = mt.SequenceShiftDataset("data/processed/multitask_tensors/tst", tokenizer.pad_idx, tokenizer.SEP_IDX, tokenizer.END_IDX)
valdl = torch.utils.data.DataLoader(valds, batch_size=248, shuffle=True, prefetch_factor=100, num_workers=20)

trainer = Trainer(model)\
    .set_trn_iterator(itertools.cycle(enumerate(trndl)))\
    .set_validation_dataloader(valdl)\
    .set_mask_percent(0.0)\
    .set_metrics_file(pathlib.Path("metrics/multitask_loss.tsv"))\
    .set_values_mask_flag(False)

trainer.start()