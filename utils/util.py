

def to_cuda_tensors(data):
    # move detection branch parameters to gpu
    device = 'cuda'
    img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx = data
    image = image.to(device)
    score_map = score_map.to(device)
    geo_map = geo_map.to(device)
    angle_map = angle_map.to(device)
    training_mask = training_mask.to(device)
    #bbox_to_img_idx = bbox_to_img_idx.to(device)
    data = (img_id, image, boxes, texts, score_map, geo_map, angle_map, training_mask, bbox_to_img_idx)
    return data

