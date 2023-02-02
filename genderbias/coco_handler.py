# TODO: make coco_img_path optional, remove requirement for COCO and use ijson streaming to deal with large caption
import os
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import CocoCaptions

# TODO: make change word banks get called when word bank edited
class COCODataLoader:
    def __init__(self, coco_ann_path, coco_img_path, male_word_bank=None, female_word_bank=None):
        self.dataset = CocoCaptions(root=coco_img_path, annFile=coco_ann_path)
        self.coco = self.dataset.coco
        self.img_ids = self.coco.getImgIds()
        self.key_male = "m"
        self.key_female = "f"
        self.key_neutral = "n"
        self.key_both = "b"
        if male_word_bank:
            self.male_word_bank = set(male_word_bank)
        else:
            self.male_word_bank = {"man", "men", "male", "boy", "gentle-man", "father", "brother", "son", "husband",
                                   "boyfriend"}
        if female_word_bank:
            self.female_word_bank = set(female_word_bank)
        else:
            self.female_word_bank = {"woman", "women", "female", "girl", "lady", "mother",
                                     "mom", "sister", "daughter", "wife", "girlfriend"}
        self.gender_table = self.change_word_banks()

    def get_image(self, img_id):
        path = self.coco.loadImgs([img_id])[0]['file_name']
        return Image.open(os.path.join(self.dataset.root, path)).convert('RGB')

    def get_image_gender(self, img_id):
        return self.gender_table[img_id]

    def get_male_ids(self):
        return self.gender_table.index[self.gender_table['gender'] == self.key_male]

    def get_female_ids(self):
        return self.gender_table.index[self.gender_table['gender'] == self.key_female]

    def get_remainder_ids(self):
        return self.gender_table.index[((self.gender_table['gender'] == self.key_both)
                                       | (self.gender_table['gender'] == self.key_neutral))]

    def change_word_banks(self):
        self.gender_table = None
        gender_table = pd.DataFrame(columns=['image_id', 'gender'], index=range(len(self.img_ids)))
        for i, img in enumerate(self.img_ids):
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img]))
            gender = 'n'

            break_loop = False
            for ann in anns:
                for word in ann['caption'].split():
                    if word in self.male_word_bank:
                        gender = 'm'
                        break_loop = True
                        break
                if break_loop:
                    break
            break_loop = False
            for ann in anns:
                for word in ann['caption'].split():
                    if word in self.female_word_bank:
                        # captions contain both masculine and feminim words so we define it both
                        if gender == 'm':
                            gender = 'b'
                        else:
                            gender = 'f'
                        break_loop = True
                        break
                if break_loop:
                    break

            gender_table.loc[i, :] = [img, gender]
        gender_table = gender_table.set_index('image_id')
        return gender_table
