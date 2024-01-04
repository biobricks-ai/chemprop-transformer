import importlib, inspect, itertools
import pathlib, signal, random, numpy as np, tqdm, sys
import cvae.models, cvae.models, cvae.tokenizer, cvae.utils
import torch, torch.utils.data, torch.nn.functional as F, torch.optim as optim

DEVICE = torch.device(f'cuda:0')

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)

# load data ===========================================================================
importlib.reload(cvae.tokenizer.selfies_property_val_tokenizer)
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data/processed/selfies_property_val_tokenizer')

class SequenceShiftDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.data = []
        self.cumulative_lengths = [0]
        cumulative_length = 0

        # file_path = next(pathlib.Path(path).glob("*.pt"))
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt")):
            file_data = torch.load(file_path)
            self.data.extend([(file_data['selfies'], file_data['assay_vals'])])
            cumulative_length += file_data['selfies'].size(0)
            self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        
        # Find which section this index falls into and update the index to be relative to that section
        file_idx = next(i for i, total_length in enumerate(self.cumulative_lengths) if total_length > idx) - 1
        idx -= self.cumulative_lengths[file_idx]
        
        idxdata = self.data[file_idx]
        selfies, raw_assay_vals = idxdata[0][idx], idxdata[1][idx]
        
        # assay_val munging - unpad, randomly permute, add sos/eos tokens
        assay_vals = raw_assay_vals[raw_assay_vals != tokenizer.pad_idx][1:-1]
        reshaped_av = assay_vals.reshape(assay_vals.size(0) // 2, 2)
        av_shuffled = reshaped_av[torch.randperm(reshaped_av.size(0)),:].reshape(assay_vals.size(0))
        
        # truncate to 59 random features
        av_truncate = av_shuffled[0:118]
        
        # add start and end tokends and pad to 120 length
        av_sos_eos = torch.cat([torch.LongTensor([tokenizer.SEP_IDX]), av_truncate, torch.LongTensor([tokenizer.END_IDX])])
        av_pad = F.pad(av_sos_eos, (0, 120 - av_sos_eos[:120].size(0)), value=tokenizer.pad_idx)
        
        # create sequence input by stacking selfies + assay_vals and 
        inp = torch.hstack([selfies, av_pad])
        
        # pad by allowing 120 selfies tokens and 60 assays
        out = torch.hstack([inp[1:], torch.tensor([tokenizer.pad_idx])])
        
        return inp, out

class Trainer():
    
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=1e-3)
        self.lossfn = cvae.models.multitask_transformer.MultitaskTransformer.loss
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=21, factor=0.9, verbose=True, min_lr=1e-6)
        self.scheduler_loss = []
        self.scheduler_loss_interval = 100
        
        self.evaluation_interval = 10000
        self.savepath = "brick/mtransform2"
        self.test_losses = [np.inf]
        self.best_test_loss = np.inf
        self.metrics_path = pathlib.Path("metrics/vaeloss.tsv")
        
    def _evaluate_and_save(self, valdl):
        model.eval()
        epochloss = 0
        for _, (inp, out) in tqdm.tqdm(enumerate(valdl), total=len(valdl)):
            if signal_received: return
            inp, out = inp.to(DEVICE), out.to(DEVICE)
            inp[:,122:240:2] = tokenizer.pad_idx
            pred, z, zvar = self.model(inp)
            loss, rec_loss, kl_loss = self.lossfn(pred.permute(0,2,1), out, z, zvar, tokenizer.pad_idx)
            epochloss += loss.item()
        
        self.test_losses.append(epochloss)
        if epochloss < self.best_test_loss:
            self.best_test_loss = epochloss
            model.module.save(cvae.utils.mk_empty_directory(self.savepath, overwrite=True))
    
    def _epochs_since_improvement(self):
        return len(self.test_losses) - np.argmin(self.test_losses)
    
    def _train_batch(self, inp, out):
        self.optimizer.zero_grad()    
        inp[:, 122:240:2] = tokenizer.pad_idx  # Mask assay values
            
        # outputs and loss
        pred, z, zvar = self.model(inp)
        loss, rec_loss, kl_loss = self.lossfn(pred.permute(0,2,1), out, z, zvar, tokenizer.pad_idx)
        
        # update model
        loss.backward()
        self.optimizer.step()
            
        return {"loss": loss.item(), "rec_loss": rec_loss.item(), "kl_loss": kl_loss.item()}

    def _write_metrics(self, epoch, mean_l, mean_rl, mean_kl):
        with open(self.metrics_path, "a") as f:
            _ = f.write(f"{epoch}\t{mean_l}\t{mean_rl}\t{mean_kl}\n")
            
    def train(self, trn_dl, val_dl):
        self.metrics_path.write_text("epoch\tloss\trecloss\tkloss\n")
        trn_dl = itertools.cycle(enumerate(trn_dl))
        trn_loss = {"loss": [], "rec_loss": [], "kl_loss": []}
        epoch = 0
        
        while self._epochs_since_improvement() < 21:
            if signal_received: return
            
            i, (inp, out) = next(trn_dl)
            lossdict = self._train_batch(inp.to(DEVICE), out.to(DEVICE))
            for k in trn_loss:
                trn_loss[k].append(lossdict[k])

            # SCHEDULER UPDATE 
            if (i + 1) % self.scheduler_loss_interval == 0:
                mean_l, mean_rl, mean_kl = [np.mean(trn_loss[k]) for k in ["loss", "rec_loss", "kl_loss"]]
                self.scheduler.step(mean_l)
                self._write_metrics(epoch, mean_l, mean_rl, mean_kl)
                print(f"epoch: {epoch}\titer: {(i+1) // self.scheduler_loss_interval}\tloss:{mean_l:.4f}\trecloss:{mean_rl:.4f}\tkl:{mean_kl:.4f}\tlr:{self.optimizer.param_groups[0]['lr']}")
                trn_loss = {"loss": [], "rec_loss": [], "kl_loss": []}
            
            # EVALUATION UPDATE
            if (i + 1) % self.evaluation_interval == 0:
                print('evaluating...')
                epoch += 1
                self._evaluate_and_save(val_dl)
                print(f"epoch: {epoch}\t eval_loss: {self.best_test_loss:.4f}")
        
        
importlib.reload(cvae.models)
model = cvae.models.MultitaskTransformer.load("brick/mtransform1").to(DEVICE)
model = torch.nn.DataParallel(model)

trnds = SequenceShiftDataset("data/processed/multitask_tensors/trn")
trndl = torch.utils.data.DataLoader(trnds, batch_size=128, shuffle=False, prefetch_factor=100, num_workers=20)
valds = SequenceShiftDataset("data/processed/multitask_tensors/tst")
valdl = torch.utils.data.DataLoader(valds, batch_size=128, shuffle=False, prefetch_factor=100, num_workers=20)
trainer = Trainer(model)
trainer.train(trndl, valdl)

# EVALUATE ===================================================================================================
import pandas as pd, inspect
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix

importlib.reload(cvae.models.multitask_transformer)
model = cvae.models.MultitaskTransformer.load("brick/mtransform2").to(DEVICE)

def extract_ordered_assays(tensor):
    assay_indexes = tokenizer.assay_indexes()
    index_assays = {v: k for k, v in assay_indexes.items()}
    return [index_assays[x.item()] for x in tensor if x.item() in assay_indexes.values()]

def extract_ordered_values(tensor):
    value_indexes = tokenizer.value_indexes()
    index_values = {v: k for k, v in value_indexes.items()}
    return [index_values[x.item()] for x in tensor if x.item() in value_indexes.values()]

def extract_probabilities_of_one(out, probs):
    # get the index of each assay token in the out tensor
    npout = out.cpu().numpy()
    out_value_sequence_indexes = np.where(np.isin(npout, list(tokenizer.value_indexes().values())))[0]
    
    value_indexes = list(tokenizer.value_indexes().values())
    assay_value_probs = probs[out_value_sequence_indexes][:,value_indexes]
    normalized_probs = assay_value_probs / assay_value_probs.sum(axis=1, keepdims=True)
    one_probs = normalized_probs[:,1]
    
    return one_probs

def evaluate(model):

    tst = SequenceShiftDataset("data/processed/multitask_tensors/tst")
    
    out_df = pd.DataFrame()
    for i in tqdm.tqdm(range(10000)):
        inp, out = tst[i]
        
        # mask all values in the input
        inp[122:240:2] = tokenizer.pad_idx
        
        probs = torch.softmax(model(inp.unsqueeze(0).to(DEVICE), temp=0.0)[0][0], dim=1).detach()
        
        probs = extract_probabilities_of_one(out, probs).cpu().numpy()
        assays = extract_ordered_assays(out)
        values = extract_ordered_values(out)
        
        i_out_df = pd.DataFrame({"i":i, "assay": assays, "value": values, "prob_1": probs })
        out_df = pd.concat([out_df, i_out_df])
    
    # out_df has columns: i, assay, value, prob_0
    # build AUC, sensitivity, specificity, accuracy, balanced_accuracy
    
    y_true, y_pred = out_df['value'].values, out_df['prob_1'].values
    
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred > 0.5)
    bac = balanced_accuracy_score(y_true, y_pred > 0.5)
    
    print(f"AUC: {auc:.4f}\tACC: {acc:.4f}\tBAC: {bac:.4f}")