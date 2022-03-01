import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class PretrainSCMBertDataGenerator(Dataset):
    def __init__(self, jsonfile_path):
        super(PretrainSCMBertDataGenerator, self).__init__()
        self.data_list = []
        with open(jsonfile_path, "r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                x = json.loads(line)
                A, B, C = x["A"], x["B"], x["C"]
                if "label" in x.keys():
                    label = int(x["label"] == "C")
                else:
                    label = 0
                self.data_list.append([A, B, C, label])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        A, B, C, label = self.data_list[index]
        return A, B, C, label

class Collate:
    def __init__(self, vocab_path, max_seq_len):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        A_batch, B_batch, C_batch, label_batch = zip(*batch)
        assert len(A_batch) == 1 
        # print("A_batch: ", A_batch)
        A_batch = self.tokenizer(A_batch[0], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        B_batch = self.tokenizer(B_batch[0], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        C_batch = self.tokenizer(C_batch[0], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
        label_batch = torch.tensor(label_batch[0])
        return A_batch, B_batch, C_batch, label_batch

def build_pretrain_scmbertx_dataloader(jsonfile_path, vocab_path, batch_size, shuffle=True, num_workers=0, max_seq_len=512):
    data_generator = PretrainSCMBertDataGenerator(jsonfile_path)
    collate = Collate(vocab_path, max_seq_len)
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )