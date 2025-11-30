import pickle
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from model_parser import parser
from dataset_utils import MyDataset
from model import CasAD
from loss_utils import MSLELoss, EarlyStopping, MAPELoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
device = torch.device(f"cuda:{args.cu}" if torch.cuda.is_available() else "cpu")

set_seed(args.seed)

def process_batch(input, labels, time_steps, fnode, nnode, norm):
    cas_time = time_steps.to(torch.float32).to(device)
    
    input = input.to(torch.float32).to(device)
    labels = labels.to(device).reshape([-1, 1])
    
    fnode = fnode.to(torch.float32).to(device)
    nnode = nnode.to(torch.float32).to(device)
    
    input = norm(input)

    return input, labels, cas_time, fnode, nnode

def main():
    data_start_time = time.time()

    with open(args.input + 'train.pkl', 'rb') as ftrain:
        _, train_tslices, train_global, train_fnode, train_newnodes, train_label = pickle.load(ftrain)
    with open(args.input + 'val.pkl', 'rb') as fval:
        _, val_tslices, val_global, val_fnode, val_newnodes, val_label = pickle.load(fval)
    with open(args.input + 'test.pkl', 'rb') as ftest:
        _, test_tslices, test_global, test_fnode, test_newnodes, test_label = pickle.load(ftest)

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_generator = MyDataset(train_tslices, train_global, train_fnode, train_newnodes, train_label, args.max_seq)
    val_generator = MyDataset(val_tslices, val_global, val_fnode, val_newnodes, val_label, args.max_seq)
    test_generator = MyDataset(test_tslices, test_global, test_fnode, test_newnodes, test_label, args.max_seq)

    train_loader = DataLoader(train_generator, batch_size=args.b_size, shuffle=True, num_workers=4, 
                              worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_generator, batch_size=args.b_size, num_workers=2, 
                           worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(test_generator, batch_size=args.b_size, num_workers=2, 
                            worker_init_fn=worker_init_fn, generator=g)

    data_end_time = time.time()
    print(f'time: {(data_end_time - data_start_time) / 60:.3f} mins')
    
    model = CasAD(args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_fn = MSLELoss()
    norm = nn.BatchNorm1d(args.max_seq).to(device)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.spath)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params}")

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f'\n=== EPOCH [{epoch+1}/{args.epochs}] ===')
        
        epoch_start_time = time.time()
        model.train()
        total = total_loss = 0
        
        for step, (input, labels, time_steps, fnode, nnode) in enumerate(train_loader):
            input, labels, cas_time, fnode, nnode = process_batch(input, labels, time_steps, fnode, nnode, norm)

            optimizer.zero_grad()
            pred_graph = model(input, cas_time, fnode, nnode)
            
            loss = loss_fn(pred_graph, labels)
            
            if step % 50 == 0:
                print(f"\nBatch {step} - Loss: {loss.item():.4f}")                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = total_loss / total
        epoch_time = time.time() - epoch_start_time
        
        print(f"\ntrain loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1} time: {epoch_time:.2f}s ({epoch_time/60:.2f}mins)")
            
        model.eval()
        total = total_loss = 0
        
        with torch.no_grad():
            for step, (input, labels, time_steps, fnode, nnode) in enumerate(val_loader):
                input, labels, cas_time, fnode, nnode = process_batch(input, labels, time_steps, fnode, nnode, norm)
                
                pred_graph = model(input, cas_time, fnode, nnode)
                
                loss = loss_fn(pred_graph, labels)
                
                if step % 50 == 0:
                    print(f"\nValidation Batch {step} - MSLE Loss: {loss.item():.4f}")
                
                total_loss += loss.item() * labels.size(0)
                total += labels.size(0)

        val_loss = total_loss / total
        print(f"\nval loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load(args.spath))
    model.eval()
    
    total = total_loss = 0
    
    with torch.no_grad():
        for step, (input, labels, time_steps, fnode, nnode) in enumerate(test_loader):
            input, labels, cas_time, fnode, nnode = process_batch(input, labels, time_steps, fnode, nnode, norm)
            
            pred_graph = model(input, cas_time, fnode, nnode)
            loss = loss_fn(pred_graph, labels)
            
            
            if step % 50 == 0:
                print(f"\nTest Batch {step} - MSLE Loss: {loss.item():.4f}")
            
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    test_loss = total_loss / total
    print(f"\ntest loss: {test_loss:.4f}")
    print(f'time: {(time.time() - start_time) / 60:.3f} mins')

if __name__ == '__main__':
    main()
