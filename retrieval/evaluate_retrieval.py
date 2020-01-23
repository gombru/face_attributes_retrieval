# Evaluate P@10 retrieval
# A result is considered relevant if it has the attribute specified by the modifier
# Then it contributes with its cosine similarity with the query sample to the P@5

import torch
import torch.nn.functional as F
import torch.nn as nn
import operator
import random
from shutil import copyfile
import os
import json
import numpy as np
import sys
sys.path.insert(1, '../model')
import model

dataset = '/home/Imatge/ssd2/CelebA/'
model_name = 'CelebA_retrieval_40mMultiHotAttributes_L2norm_HardNegatives_2ndTr_lr00001_epoch_3_ValLoss_0.008.pth'
model_name = model_name.strip('.pth')
split = 2  # 2 is test
embedding_dimensionality = 300

attribute_mult = 1

queries_file = 'queries.txt'

gpus = [1]
gpu = 1
gpu_cuda_id = 'cuda:' + str(gpu)
embedding_dim = 300
num_results = 10

distance_norm = 2 # 2
dist = nn.PairwiseDistance(p=distance_norm)
print("Using pairwise distance with norm: " + str(distance_norm))
normalize = True # Normalize img embeddings and attributes embeddings using L2 norm
print("L2 normalize img and query embeddings: " + str(normalize))

print("Reading imgs embeddings ...")
img_embeddings_path = dataset + 'img_embeddings/' + model_name + '/img_embeddings_test.json'
img_embeddings = json.load(open(img_embeddings_path))
print("Putting image embeddings in a tensor")
img_embeddings_tensor = torch.zeros([len(img_embeddings), embedding_dim], dtype=torch.float32).cuda(gpu)
img_ids = []
for i, (img_id, img_embedding) in enumerate(img_embeddings.items()):
    img_ids.append(img_id)
    img_np_embedding = np.asarray(img_embedding, dtype=np.float32)
    # if normalize:  # This normalization is already done in the model, so already done in the precomputed img embeddings
    #     img_np_embedding /= np.linalg.norm(img_np_embedding)
    img_embeddings_tensor[i, :] = torch.from_numpy(img_np_embedding).cuda(gpu)
del img_embeddings

attributes = {}
print("Loading attributes")
for i, line in enumerate(open(dataset + 'Anno/' + 'list_attr_celeba.txt')):
    d = line.split(' ')
    img_id = d[0]
    # if img_id not in self.img_ids: # Img does not belogn to this split # Not doing it because it's slow, so we are duplicating attributes in memory for each split
    #     continue
    d = list(filter(None, d)) # Filter out empty elements (double spaces)
    if 'jpg' not in d[0]: continue
    img_attributes = np.array(d[1:])
    img_attributes = img_attributes.astype(np.float32)
    img_attributes[img_attributes < 0] = 0 # Change -1 attributes to 0 
    attributes[img_id] = img_attributes
print("Attributes loaded")

# Load attribute indices
attribute_indices = {}
for line in open(dataset + 'attribute_indices.txt'):
    d = line.split(' ')
    attribute_indices[d[0]] = int(d[1])

# Load the model to compute query embeddings
state_dict = torch.load(dataset + 'models/saved/' + model_name + '.pth.tar',
                        map_location={'cuda:0':gpu_cuda_id, 'cuda:1':gpu_cuda_id, 'cuda:2':gpu_cuda_id, 'cuda:3':gpu_cuda_id})
model_test = model.AttModelTest(embedding_dimensionality)
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)
model_test.eval()

total_p10 = 0
total_queries = 0
total_correct_att = 0
total_cosine_sim = 0

for i,line in enumerate(open(queries_file)):
    query_p10 = 0
    d = line.split(' ')
    query_img_name = d[0]
    if 'jpg' not in query_img_name: 
        continue
    query_img_idx = img_ids.index(query_img_name)
    operation = d[1][0]  # +/- This only supports the add or substract of a single attribute, and always gives the same weight to the image and to the attribute. 
    
    # Get query attribute representation
    query_attribute_name = d[1][1:].replace('\n','') 
    query_attribute_idx = attribute_indices[query_attribute_name]
    query_attribute_representation = np.zeros(40, dtype=np.float32)
    query_attribute_representation[query_attribute_idx] = 1
    query_attribute_representation_tensor = torch.from_numpy(query_attribute_representation)
    
    # Get query attribute embedding
    with torch.no_grad():
        query_attribute_representation_tensor = query_attribute_representation_tensor.unsqueeze(0)
        query_attribute_representation_tensor = torch.autograd.Variable(query_attribute_representation_tensor)
        query_attribute_embedding = model_test(query_attribute_representation_tensor)

    # Compute query embedding
    query_attribute_embedding *= attribute_mult
    if operation == '+':
        query_value = 1
        query_embedding = img_embeddings_tensor[query_img_idx,:] + query_attribute_embedding 
    elif operation == '-':
        query_value = 0
        query_embedding = img_embeddings_tensor[query_img_idx,:] - query_attribute_embedding
    else:
        print("Operation " + str(operation) + " not supported.")
        continue

    if normalize: # Here I need to re-normalize to get the same norm after the operation between embeddings
        norm = query_embedding.norm(p=2, dim=1, keepdim=True)
        query_embedding = query_embedding.div(norm)


    distances = dist(img_embeddings_tensor, query_embedding)
    results_indices_sorted = np.array(distances.sort(descending=False)[1][0:num_results].cpu())

    # Get results attributes GT
    query_img_att = attributes[query_img_name]
    for result_idx in results_indices_sorted:
        result_img_GT_att = attributes[img_ids[result_idx]]
        if result_img_GT_att[query_attribute_idx] == query_value: # Check if result has the modified attribute
            total_correct_att += 1
            # Add cosine similarity to query P@10
            dot = np.dot(query_img_att, result_img_GT_att)
            norma = np.linalg.norm(query_img_att)
            normb = np.linalg.norm(result_img_GT_att)
            cos = dot / (norma * normb)
            query_p10 += cos
            total_cosine_sim += cos

    total_p10 += query_p10 / 10
    total_queries+=1

    # Save resulting IMG
    out_folder = dataset + '/retrieval_results/' + model_name + '/' + operation + query_attribute_name + '/' + query_img_name.replace('.jpg','') + '/'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    # Copy query image
    copyfile(dataset + 'img_resized/' + query_img_name, out_folder + 'query_img_' + query_img_name)
    # Copy results
    for i,result_idx in enumerate(results_indices_sorted):
        copyfile(dataset + 'img_resized/' + img_ids[result_idx], out_folder + img_ids[result_idx].replace('.jpg','') + '_' + str(i) + '.jpg')


total_p10 /= total_queries
print("P@10: " + str(total_p10))
print("Total Correct att: " + str(total_correct_att))
print("Num queries: " + str(total_queries))
print("Mean correct att per query: " + str(float(total_correct_att) / total_queries))
print("Mean cosine sim of correct att: " + str(float(total_cosine_sim) / total_correct_att))