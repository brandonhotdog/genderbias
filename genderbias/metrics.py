import numpy as np
import torch
from utils.helper import GetTextQueries
from utils.banks import text_queries_bias_at_k
from tqdm import tqdm

class GenderBlindness:
    def __init__(self, coco_dataloader, text_queries=["a photo of a man", "a photo of a woman"]):
        self.dataloader = coco_dataloader
        self.text_queries = text_queries

    def test(self, model, samples=10):
        pass


class BiasAtK:
    def __init__(self, coco_dataloader, text_queries_path=None):
        self.dataloader = coco_dataloader
        if text_queries_path:
            self.text_queries = GetTextQueries(text_queries_path)
        else:
            self.text_queries = text_queries_bias_at_k

    def test(self, model, k=[10], samples=10):
        if isinstance(k, int):
            k = [k]
        male_ids = self.dataloader.get_male_ids()
        female_ids = self.dataloader.get_female_ids()
        remainder_ids = self.dataloader.get_remainder_ids()
        pbar = tqdm(total=len(male_ids)+len(female_ids)+len(remainder_ids))

        text_features = model.encode_text(self.text_queries)
        male_logits = model.get_logits_tensor(male_ids, text_features, pbar, self.dataloader)
        female_logits = model.get_logits_tensor(female_ids, text_features, pbar, self.dataloader)
        remainder_logits = model.get_logits_tensor(remainder_ids, text_features, pbar, self.dataloader)

        bias_at_k = np.zeros((len(k), len(text_features)))
        maxk = max(k)
        class_min = min(len(male_ids), len(female_ids))
        if len(male_ids) == len(female_ids):
            samples = 1
        for _ in range(samples):
            sample_male_logits = male_logits[np.random.choice(np.arange(male_logits.shape[0]), replace=False, size=class_min),:]
            sample_female_logits = female_logits[np.random.choice(np.arange(female_logits.shape[0]), replace=False, size=class_min),:]
            logits_matrix = torch.concat([sample_male_logits, sample_female_logits, remainder_logits])
            topk = torch.topk(logits_matrix, dim=0, k=maxk, sorted=True)
            results_matrix = topk.indices.cpu().numpy()
            for col_id, column in enumerate(results_matrix.T):
                men = 0
                women = 0
                k_count = 0
                for i in range(maxk):
                    if column[i] < class_min:
                        men += 1
                    elif column[i] < (2*class_min):
                        women += 1
                    if (i+1) in k:
                        result = 0
                        if not (men == 0 and women == 0):
                            result = (men - women) / (men + women)
                        bias_at_k[k_count, col_id] += result/float(samples)
                        k_count += 1
        return np.average(bias_at_k, axis=1), bias_at_k