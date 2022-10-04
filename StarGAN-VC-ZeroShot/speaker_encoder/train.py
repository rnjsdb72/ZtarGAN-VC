from tqdm import tqdm
import json
import argparse
from collections import namedtuple

import torch

def load_checkpoint():
    pass

def train(model, train_loader, test_loader, optimizer, loss, config):
    print('Start to Train,,,')
    start_epoch = 0
    best_loss = 0
    
    if config.resume_from:
        model, optimizer, scheduler, start_epoch, best_loss = load_checkpoint()
        
    outer_bar = tqdm(total=config.train.step, desc='Training', position=0)
    outer_bar.n = start_epoch
    outer_bar.update()
    for epoch in outer_bar:
        model.train()
        
        running_loss = 0
        sum_loss = 0
        
        inner_bar = tqdm(enumerate(train_loader), total=len(train_loader), position=1)
        for step, input in inner_bar:
            # x = x.to(device)
            # y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss_ = loss(output, y)
            
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            running_loss = sum_loss / (step+1)
            
            description =  f'Epoch [{epoch+1}/{config.train.step}], Step [{step+1}/{len(train_loader)}]: ' 
            description += f'running Loss: {round(running_loss,4)}'
            inner_bar.set_description(description)
            
            # log
            #
            
            # lr 조정
            #
            
            # validation
            #

def main(config):
    # load data
    
    # model, optimizer, loss
    
    # train
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgs', type=str)
    args = parser.parse_args()
    with open(args.cfgs, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys()))(*d.values())
    
    main(cfgs)