from torch.utils.data import Dataset


class ToxicRankDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self._data=data
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self):
        return len(self._data)
    def __getitem__(self, index):
        tokenized = self.tokenizer(text=self._data['text'].values[index],
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        
        if 'label' not in self._data.columns:
          return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

        target = self._data['label'].values[index]
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze(), target