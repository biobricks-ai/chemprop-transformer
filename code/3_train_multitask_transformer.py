# # TODO: make a good histogram of the AUCs for each property.
# TODO:Show how AUC change over time. Median AUC for each position with line chart./Another way, for each property, it improves over time.
# TODO:what's the difference between 0 to 1, 1 to 2, 2 to 3, take the median difference.
# TODO: New training for adding inp
import numpy as np
import itertools, pathlib, numpy as np, tqdm, inspect
import torch, torch.utils.data, torch.optim as optim
from torch.optim import lr_scheduler
import cvae.tokenizer, cvae.utils as utils
import cvae.models.multitask_transformer as mt 

DEVICE = torch.device(f'cuda:0')

# load data ===========================================================================
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data2/processed/selfies_property_val_tokenizer')

class Trainer():
    
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(model.parameters(),lr=0.1,betas = (0.9, 0.98), eps=1e-9)
        self.lossfn = mt.MultitaskTransformer.lossfn(ignore_index=tokenizer.pad_idx)
        # self.lossfn = mt.LabelSmoothingCrossEntropySequence(ignore_index=tokenizer.pad_idx)
        
        # sched_args = {"patience":2, "factor":0.9, "verbose":True, "min_lr":1e-7, "cooldown":5 }
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **sched_args)
        
        # T_0 = 10  # Number of epochs for the first cycle
        # T_mult = 2  # A factor that increases T_0 for each subsequent cycle, 1 means constant cycle length
        # eta_min = 1e-7  # Minimum learning rate
        # self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1)
        
        self.scheduler = mt.NoamLR(self.optimizer, model_size=model.module.hdim, warmup_steps=4000)
        
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
    
    def _evaluation_loss(self, valdl):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0  # Keep track of the total count of values considered for accuracy

        value_indexes = torch.LongTensor(list(tokenizer.value_indexes().values())).to(DEVICE)

        for _, (inp, teach, out) in tqdm.tqdm(enumerate(valdl), total=len(valdl)):
            inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
            with torch.no_grad():  # No need to track gradients during evaluation
                pred = self.model(inp, teach)
                loss = self.lossfn(self.model.parameters(), pred.permute(0,2,1), out)
                # loss = self.lossfn(pred, out)

                # Compute softmax probabilities
                prob = torch.softmax(pred, dim=2)

                # Filter out the relevant outputs and predictions for accuracy calculation
                mask = torch.isin(out, value_indexes)
                outval = torch.masked_select(out, mask)
                prbval = torch.masked_select(torch.argmax(prob, dim=2), mask)

                # Compute accuracy for this batch and accumulate
                acc = torch.sum(outval == prbval).float()
                total_acc += acc
                total_count += outval.size(0)  # Update the total count

                # Accumulate loss
                total_loss += loss.item() * inp.size(0)  # Multiply by batch size to later compute the correct mean

        # Compute mean loss and accuracy
        mean_loss = total_loss / len(valdl.dataset)
        mean_acc = total_acc / total_count if total_count > 0 else torch.tensor(0.0)

        return mean_loss, mean_acc.item()  # Return mean accuracy as a Python scalar
                
    def _epochs_since_improvement(self):
        return len(self.test_losses) - np.argmin(self.test_losses)
    
    def _train_batch(self, inp, teach, out):
        _ = self.model.train()
        self.optimizer.zero_grad()
        
        # outputs and loss
        pred = self.model(inp, teach)
        loss = self.lossfn(self.model.parameters(), pred.permute(0,2,1), out)
        # loss = self.lossfn(pred,out)
        
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
            # x = torch.gt(torch.sum(torch.isin(out, value_indexes),dim=1),9)
            # inp, teach, out = inp[x],teach[x],out[x]
            # this_batch_size = inp.size(0)
            
            mask = torch.rand(inp.shape, device=DEVICE) < self.mask_percent
            mask[:,0] = False # don't mask the first token
            inp = inp.masked_fill(mask, tokenizer.pad_idx)
            
            mask = torch.rand(teach.shape, device=DEVICE) < self.mask_percent
            mask[:,0] = False # don't mask the first token
            teach = teach.masked_fill(mask, tokenizer.pad_idx)

            loss = self._train_batch(inp, teach, out)
            trn_loss.append(loss)
            self.scheduler.step()
            utils.write_path(self.metrics_path,f"train\t{i}\t{loss}\t{self.optimizer.param_groups[0]['lr']:.12f}\n")
            
            # SCHEDULER UPDATE 
            if (i + 1) % self.scheduler_loss_interval == 0:
                mean_loss = np.mean(trn_loss)
                utils.write_path(self.metrics_path,f"scheduler\t{i}\t{mean_loss}\t{self.optimizer.param_groups[0]['lr']:.12f}\n")
                trn_loss = []
                # self.scheduler.step(mean_loss)
            
            # EVALUATION UPDATE
            if (i + 1) % evaluation_interval == 0:
                epoch += 1
                eval_loss, eval_acc = self._evaluation_loss(valdl)
                self.test_losses.append(eval_loss)
                if eval_loss < self.best_test_loss:
                    self.best_test_loss = eval_loss
                    self.model.module.save(self.savepath)
                
                utils.write_path(self.metrics_path,f"eval\t{i}\t{self.test_losses[-1]}\n")
                print(f"epoch: {epoch}\teval_loss: {self.best_test_loss:.4f}\teval_acc: {eval_acc:.4f}\tLR: {self.optimizer.param_groups[0]['lr']:.12f}")

import importlib
importlib.reload(mt)
model = mt.MultitaskTransformer(tokenizer).to(DEVICE)
# model = mt.MultitaskTransformer.load("brick/mtransform2")
model = torch.nn.DataParallel(model)

trnds = mt.SequenceShiftDataset("data/processed/multitask_tensors/trn", tokenizer)
trndl = torch.utils.data.DataLoader(trnds, batch_size=512, shuffle=True, prefetch_factor=100, num_workers=20)
valds = mt.SequenceShiftDataset("data/processed/multitask_tensors/tst", tokenizer)
valdl = torch.utils.data.DataLoader(valds, batch_size=512, shuffle=True, prefetch_factor=100, num_workers=20)

trainer = Trainer(model)\
    .set_trn_iterator(itertools.cycle(enumerate(trndl)))\
    .set_validation_dataloader(valdl)\
    .set_mask_percent(0.1)\
    .set_metrics_file(pathlib.Path("metrics/multitask_loss.tsv"), overwrite=True)

trainer.start()
