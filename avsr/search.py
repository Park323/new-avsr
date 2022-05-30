def greedySearch(model, features, *args, **kwargs):
    max_len = model.config.model.max_len
    preds = torch.full((features.size(0), max_len+1), 0).to(features.device).to(int)
    pred_lengths = torch.full((features.size(0),), 0, dtype=int, device=features.device)
    active_batch = torch.full((features.size(0),), True, dtype=bool, device=features.device)
            
    loop_idx = 0
    while loop_idx <= max_len:
        # pdb.set_trace()
        # End the loop
        if active_batch.sum()==0:
            break 
        # Fill sos_id
        elif loop_idx == 0:
            preds[:,loop_idx] = 1
        # if unfinished batch exists
        else:
            targets = F.one_hot(preds[active_batch,:loop_idx], num_classes = model.vocab_size)
            targets = model.target_embedding(targets.to(torch.float32))
            outputs = model.decoder(targets, features[active_batch], train=False, pad_id=0)
            outputs = model.ceLinear(outputs)
            topk_ids = outputs.topk(k=1, dim=-1).indices[:,-1,0]
            preds[active_batch, loop_idx] = topk_ids
            
            pred_lengths[active_batch] += 1
            active_batch[preds[:, loop_idx] == 2] = False
            
        loop_idx += 1
        
    preds = F.one_hot(preds[:,1:], model.vocab_size).to(float)
    preds = F.log_softmax(preds, dim=-1)
    
    return preds, pred_lengths