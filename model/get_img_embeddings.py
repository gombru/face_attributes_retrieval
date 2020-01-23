import os
import torch.utils.data
import model
import json
import numpy as np
import CelebA_dataset


dataset = '/home/Imatge/ssd2/CelebA/'
split_test = 2

embedding_dimensionality = 300

batch_size = 700
workers = 6

model_name = 'CelebA_retrieval_40mMultiHotAttributes_L2norm_HardNegatives_2ndTr_lr00001_epoch_3_ValLoss_0.008.pth'
model_name = model_name.strip('.pth')

gpus = [3]
gpu = 3
gpu_cuda_id = 'cuda:' + str(gpu)


if not os.path.exists(dataset + 'img_embeddings/' + model_name):
    os.makedirs(dataset + 'img_embeddings/' + model_name)

output_file_path = dataset + 'img_embeddings/' + model_name + '/img_embeddings_test.json'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset + 'models/' + model_name + '.pth.tar',
                        map_location={'cuda:0':gpu_cuda_id, 'cuda:1':gpu_cuda_id, 'cuda:2':gpu_cuda_id, 'cuda:3':gpu_cuda_id})


model_test = model.ImgModelTest(embedding_dimensionality)
model_test = torch.nn.DataParallel(model_test, device_ids=gpus).cuda(gpu)
model_test.load_state_dict(state_dict, strict=False)

test_dataset = CelebA_dataset.CelebA_dataset(dataset, split_test, mirror=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                          pin_memory=True)

img_embeddings = {}

with torch.no_grad():
    model_test.eval()
    for i, (img_name, img, att_p, att_n) in enumerate(test_loader):
        img = torch.autograd.Variable(img)
        outputs = model_test(img)

        for batch_idx,img_embedding in enumerate(outputs):
            img_embeddings[str(img_name[batch_idx])] = np.array(img_embedding.cpu()).tolist()
        print(str(i) + ' / ' + str(len(test_loader)))

print("Writing results")
json.dump(img_embeddings, output_file)
output_file.close()

print("DONE")