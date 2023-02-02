import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


class GenericModel:
    def __init__(self):
        self.image_embedding_df = None
        pass

    def encode_text(self, text_queries):
        raise NotImplementedError

    def encode_image(self, image):
        raise NotImplementedError

    def get_logits(self, image, text_embeddings):
        raise NotImplementedError

    def load_image_embeddings(self, path):
        self.image_embedding_df = pd.read_csv(path, index_col=0)

    def save_image_embeddings(self, path):
        self.image_embedding_df.to_csv(path)

    def add_image_embedding(self, image_embedding, img_id):
        if isinstance(self.image_embedding_hold, int):
            self.image_embedding_hold = torch.zeros((self.image_embedding_hold, image_embedding.shape[1]))
        i = len(self.image_embedding_ids_hold)
        self.image_embedding_hold[i, :] = image_embedding
        self.image_embedding_ids_hold.append(img_id)

    def update_image_embeddings_df(self):
        # If no image embeddings were made then no need to update df
        if len(self.image_embedding_ids_hold) == 0:
            self.image_embedding_hold = None
            return None
        self.image_embedding_hold = self.image_embedding_hold[:len(self.image_embedding_ids_hold)]
        df = pd.DataFrame(index=self.image_embedding_ids_hold, data=self.image_embedding_hold.cpu().detach().numpy())
        if self.image_embedding_df is None:
            self.image_embedding_df = df
        else:
            df.columns = self.image_embedding_df.columns
            self.image_embedding_df = pd.concat([self.image_embedding_df, df])
        self.image_embedding_hold = None
        self.image_embedding_ids_hold = None

    def get_image_features(self, img_id, dataloader=None):
        if self.image_embedding_df is not None and img_id in self.image_embedding_df.index:
            return torch.tensor(self.image_embedding_df.loc[img_id], dtype=torch.float)
        else:
            embedded_image = self.encode_image(dataloader.get_image(img_id)).cpu().detach()
            self.add_image_embedding(embedded_image, img_id)
            return embedded_image

    def get_logits_tensor(self, img_ids, text_embeddings, pbar, dataloader=None):
        self.image_embedding_hold = len(img_ids)
        self.image_embedding_ids_hold = []

        logits_tensor = torch.zeros((len(img_ids), len(text_embeddings)))
        for i, img in enumerate(img_ids):
            logits_tensor[i, :] = self.get_logits(self.get_image_features(img, dataloader), text_embeddings.cpu().detach())
            pbar.update(1)

        self.update_image_embeddings_df()
        return logits_tensor