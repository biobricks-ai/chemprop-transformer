import importlib, inspect, itertools
import pathlib, signal, random, numpy as np, tqdm, sys
import cvae.models, cvae.models, cvae.tokenizer, cvae.utils as utils
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
        self.lossfn = cvae.models.multitask_transformer.MultitaskTransformer.lossfn()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=21, factor=0.9, verbose=True, min_lr=1e-6)
        self.scheduler_loss = []
        self.scheduler_loss_interval = 100
        
        self.evaluation_interval = 1000
        self.savepath = "brick/mtransform2"
        self.test_losses = [np.inf]
        self.best_test_loss = np.inf
        self.metrics_path = pathlib.Path("metrics/multitask_loss.tsv")
        
    def _evaluation_loss(self, valdl):
        self.model.eval()
        epochloss = []
        for _, (inp, out) in tqdm.tqdm(enumerate(valdl), total=len(valdl)):
            if signal_received: return
            inp, out = inp.to(DEVICE), out.to(DEVICE)
            inp[:,122:240:2] = tokenizer.pad_idx
            pred = self.model(inp)
            loss = self.lossfn(pred.permute(0,2,1), out)
            epochloss.append(loss.item())
        
        return np.mean(epochloss)
                
    def _epochs_since_improvement(self):
        return len(self.test_losses) - np.argmin(self.test_losses)
    
    def _train_batch(self, inp, out):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Mask assay values
        # inp[:, 122:240:2] = tokenizer.pad_idx  
        
        # Randomly mask 50% of assay values
        mask = torch.rand(inp[:, 122:240:2].shape).to(DEVICE) < 0.5
        inp[:, 122:240:2] = inp[:, 122:240:2].masked_fill(mask, tokenizer.pad_idx)
            
        # outputs and loss
        pred = self.model(inp)
        loss = self.lossfn(pred.permute(0,2,1), out)
        
        # update model
        loss.backward()
        self.optimizer.step()
            
        return loss.item()

    def train(self, trndl, valdl):
        utils.write_path(self.metrics_path,"epoch\tloss\tlearning_rate\n", mode='w')
        it_trndl = itertools.cycle(enumerate(trndl))
        trn_loss = []
        epoch = 0
        
        while self._epochs_since_improvement() < 10:
            if signal_received: return
            
            i, (inp, out) = next(it_trndl)
            loss = self._train_batch(inp.to(DEVICE), out.to(DEVICE))
            trn_loss.append(loss)
            utils.write_path(self.metrics_path,f"train\t{i}\t{loss}\t{self.optimizer.param_groups[0]['lr']:.4f}\n")
            
            # SCHEDULER UPDATE 
            if (i + 1) % self.scheduler_loss_interval == 0:
                mean_loss = np.mean(trn_loss)
                self.scheduler.step(mean_loss)
                utils.write_path(self.metrics_path,f"scheduler\t{i}\t{mean_loss}\t{self.optimizer.param_groups[0]['lr']:.4f}\n")
                trn_loss = []
            
            # EVALUATION UPDATE
            if (i + 1) % self.evaluation_interval == 0:
                epoch += 1
                eval_loss = self._evaluation_loss(valdl)
                self.test_losses.append(eval_loss)
                if eval_loss < self.best_test_loss:
                    self.best_test_loss = eval_loss
                    self.model.module.save(self.savepath)
                
                utils.write_path(self.metrics_path,f"eval\t{i}\t{self.test_losses[-1]}\n")
                print(f"epoch: {epoch}\t eval_loss: {self.best_test_loss:.4f}")
        
        
importlib.reload(cvae.models)
model = cvae.models.MultitaskTransformer.load("brick/mtransform1").to(DEVICE)
model = torch.nn.DataParallel(model)

trnds = SequenceShiftDataset("data/processed/multitask_tensors/trn")
trndl = torch.utils.data.DataLoader(trnds, batch_size=128, shuffle=False, prefetch_factor=100, num_workers=20)
valds = SequenceShiftDataset("data/processed/multitask_tensors/tst")
valdl = torch.utils.data.DataLoader(valds, batch_size=128, shuffle=False, prefetch_factor=100, num_workers=20)
trainer = Trainer(model)

signal_received = False
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

def extract_assay_value_dict(tensor):
    assay_indexes = torch.LongTensor(list(tokenizer.assay_indexes().values())).to(DEVICE)
    # torch where tensor value is in assay_indexes
    indexes = torch.isin(tensor, assay_indexes)
    # shift index one to right to get value indexes
    value_indexes = torch.where(indexes)[0] + 1
    tensor_assays = tensor[indexes].cpu().numpy()
    tensor_values = tensor[value_indexes].cpu().numpy()
    # zip together to get assay:value pairs
    return dict(zip(tensor_assays, tensor_values))

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
    for i in tqdm.tqdm(range(1000)):
        inp, out = tst[i]
        inp, out = inp.to(DEVICE), out.to(DEVICE)
        
        # mask all values in the input
        # inp[122:240:2] = tokenizer.pad_idx
                
        # Randomly mask 50% of assay values
        mask = torch.rand(inp[122:240:2].shape).to(DEVICE) < 0.5
        mask_inp = inp
        mask_inp[122:240:2] = inp[122:240:2].masked_fill(mask, tokenizer.pad_idx)
            
        pred = model(mask_inp.unsqueeze(0))
        probs = torch.softmax(pred[0],dim=1).detach()
        
        probs = extract_probabilities_of_one(out, probs).cpu().numpy()
        assays = extract_ordered_assays(out)
        values = extract_ordered_values(out)
        is_masked = [x == tokenizer.PAD_IDX for x in list(extract_assay_value_dict(mask_inp).values())]
        
        i_out_df = pd.DataFrame({"i":i, "assay": assays, "value": values, "is_masked": is_masked, "prob_1": probs })
        out_df = pd.concat([out_df, i_out_df])
    
    # out_df has columns: i, assay, value, prob_0
    # build AUC, sensitivity, specificity, accuracy, balanced_accuracy
    evaldf = out_df[out_df['is_masked'] == True]
    y_true, y_pred = evaldf['value'].values, evaldf['prob_1'].values
    
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred > 0.5)
    bac = balanced_accuracy_score(y_true, y_pred > 0.5)
    
    print(f"AUC: {auc:.4f}\tACC: {acc:.4f}\tBAC: {bac:.4f}")