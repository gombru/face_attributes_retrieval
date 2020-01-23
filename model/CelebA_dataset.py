import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import model
import numpy as np
from PIL import Image
import image_processing


class CelebA_dataset(Dataset):
    def __init__(self, root_dir, split, mirror):

        self.root_dir = root_dir
        self.split = split
        self.mirror = mirror

        max_elements = None

        print("Loading img ids")
        self.img_ids = []
        for i,line in enumerate(open(self.root_dir + 'Eval/' + 'list_eval_partition.txt')):
            d = line.split(' ')
            if int(d[1]) == self.split:
                self.img_ids.append(d[0])

        if max_elements:
            self.img_ids = self.img_ids[0:max_elements]
        print("Number of samples in split: " + str(len(self.img_ids)))

        self.attributes = {}

        print("Loading attributes")
        for i, line in enumerate(open(self.root_dir + 'Anno/' + 'list_attr_celeba.txt')):
            d = line.split(' ')
            img_id = d[0]
            # if img_id not in self.img_ids: # Img does not belogn to this split # Not doing it because it's slow, so we are duplicating attributes in memory for each split
            #     continue
            d = list(filter(None, d)) # Filter out empty elements (double spaces)
            if 'jpg' not in d[0]: continue
            img_attributes = np.array(d[1:])
            img_attributes = img_attributes.astype(np.float32)
            img_attributes[img_attributes < 0] = 0 # Change -1 attributes to 0 
            self.attributes[img_id] = img_attributes
        print("Attributes loaded")


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, idx):

        img_name = self.root_dir + 'img_resized/' + self.img_ids[idx]
        img = Image.open(img_name)
        if self.mirror:
            img = image_processing.Mirror(img)
        img = np.array(img, dtype=np.float32)
        img = image_processing.PreprocessImage(img)

        att_p = self.attributes[self.img_ids[idx]]

        # Negative attributes selection

        neg_type = random.randint(0,2)
        #neg_type = 0
        
        if neg_type < 2:
            # Select a random negative from the existing ones, just checking it's not equal
            while True:
                negative_img_id = random.choice(self.img_ids)
                att_n = self.attributes[negative_img_id]
                if not np.array_equal(att_p, att_n):
                    break

        else:
            # Create hard negative by changing from 1 to 3 elements of the GT attributes
            num_changes = random.randint(1,3)
            att_n = np.copy(att_p)
            for c in range(0,num_changes):
                att_idx = random.randint(0,len(att_n) - 1)
                if att_p[att_idx] == 0:
                    att_n[att_idx] = 1
                else:
                    att_n[att_idx] = 0

        # Build tensors
        img = torch.from_numpy(np.copy(img))
        att_p = torch.from_numpy(att_p)
        att_n = torch.from_numpy(att_n)

        return self.img_ids[idx], img, att_p, att_n