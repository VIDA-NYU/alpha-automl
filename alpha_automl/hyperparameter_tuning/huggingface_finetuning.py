import logging

import evaluate
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from alpha_automl._optional_dependency import check_optional_dependency

ml_task = 'nlp'
check_optional_dependency('transformers', ml_task)
from transformers import AutoImageProcessor, AutoTokenizer, Trainer, TrainingArguments

logger = logging.getLogger(__name__)


class HuggingfaceDataset(Dataset):
    def __init__(
        self, df, text_col=None, image_col=None, label_col=None, processor_model=None
    ):
        self.df = df
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.text_processor = (
            AutoTokenizer.from_pretrained(processor_model) if text_col else None
        )
        self.image_processor = (
            AutoImageProcessor.from_pretrained(processor_model) if image_col else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        labels = torch.tensor(self.df[self.label_col][idx])
        if self.image_processor:
            file_name = self.df[self.image_col][idx]
            image = Image.open(file_name).convert('RGB')
            if image.size == (1, 1):
                image = Image.new('RGB', (224, 224))
            pixel_values = self.image_processor(image, return_tensors='pt').pixel_values
            return {'labels': labels.float(), 'pixel_values': pixel_values.squeeze()}
        elif self.text_processor:
            text = self.df[self.text_col][idx]
            inputs = self.text_processor(text, padding="max_length", truncation=True, return_tensors='pt')
            inputs['labels'] = labels.float()
            inputs['input_ids'] = inputs['input_ids'].squeeze()
            inputs['token_type_ids'] = inputs['token_type_ids'].squeeze()
            inputs['attention_mask'] = inputs['attention_mask'].squeeze()
            return inputs
        else:
            return {}


class HuggingfaceFinetuner:
    def __init__(self, epochs=3):
        self.metric = evaluate.load('accuracy')
        self.training_args = TrainingArguments(
            output_dir='tmp/', evaluation_strategy='epoch', num_train_epochs=epochs,
        )
        self.train_data = None
        self.test_data = None
        self.trainer = None

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def init_dataset(
        self,
        train_df,
        test_df,
        processor_model,
        text_col=None,
        image_col=None,
        label_col=None,
    ):
        self.train_data = HuggingfaceDataset(
            train_df, text_col, image_col, label_col, processor_model
        )
        self.test_data = HuggingfaceDataset(
            test_df, text_col, image_col, label_col, processor_model
        )

    def finetune(self, model):
        if not self.train_data or not self.test_data:
            return ValueError(
                'Train_data and test_data not init yet.\nCall HuggingfaceFinetuner.init_dataset() first!'
            )

        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            compute_metrics=self.compute_metrics,
        )
        
        self.trainer.train()
        

    def save_model(self, output_path):
        if self.trainer:
            self.trainer.save_model(output_path)
        else:
            return ValueError(
                'self.trainer not finetuned yet.\nCall HuggingfaceFinetuner.finetune() first!'
            )
