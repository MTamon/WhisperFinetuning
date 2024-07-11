from finetune.utils.data_forming import load_data
import numpy as np
from datasets import Dataset, Audio

df = load_data("/content/drive/MyDrive/ja_voice_finetuning/data/voice_data.csv")

msk = np.random.rand(len(df)) < 0.7

train_dataset = (
    Dataset.from_pandas(df[msk])
    .cast_column("path", Audio(sampling_rate=16000))
    .rename_column("path", "audio")
    .remove_columns(["sampling_rate"])
)
validate_dataset = (
    Dataset.from_pandas(df[~msk])
    .cast_column("path", Audio(sampling_rate=16000))
    .rename_column("path", "audio")
    .remove_columns(["sampling_rate"])
)

################################
from datasets import DatasetDict

datasets = DatasetDict({"train": train_dataset, "validate": validate_dataset})

################################
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("kotoba-tech/kotoba-whisper-v1.0")


################################
def prepare_dataset(batch):
    audio = batch["audio"]

    # 音響特徴量抽出
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # 正解のテキストをlabel idにエンコード
    batch["labels"] = processor.tokenizer(batch["correct"]).input_ids
    return batch


################################
# prepared_datasets = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=1)
prepared_datasets = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"])

################################
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        # 音響特徴量側をまとめる処理
        # (一応バッチ単位でパディングしているが、すべて30秒分であるはず)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # トークン化された系列をバッチ単位でパディング
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # attention_maskが0の部分は、トークンを-100に置き換えてロス計算時に無視させる
        # -100を無視するのは、PyTorchの仕様
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # BOSトークンがある場合は削除
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 整形したlabelsをバッチにまとめる
        batch["labels"] = labels

        return batch


################################
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

################################
import evaluate
import spacy
import ginza

metric = evaluate.load("wer")
nlp = spacy.load("ja_ginza")
ginza.set_split_mode(nlp, "C")  # CはNEologdの意らしいです


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 分かち書きして空白区切りに変換
    pred_str = [" ".join([str(i) for i in nlp(j)]) for j in pred_str]
    label_str = [" ".join([str(i) for i in nlp(j)]) for j in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


################################
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./out/kotoba-whisper-v1.0",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-4,
    # warmup_steps=500, # Hugging Faceブログではこちら
    warmup_steps=100,
    # max_steps=4000, # Hugging Faceブログではこちら
    max_steps=30,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    # save_steps=1000, # Hugging Faceブログではこちら
    save_steps=10,
    # eval_steps=1000, # Hugging Faceブログではこちら
    eval_steps=10,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

################################
from transformers import Seq2SeqTrainer
from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained("kotoba-tech/kotoba-whisper-v1.0")

model.generation_config.language = "ja"


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=prepared_datasets["train"],
    eval_dataset=prepared_datasets["validate"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

################################
# prediction_output = trainer.predict(prepared_datasets["validate"]).metrics["test_wer"]

import pandas as pd

pd.DataFrame(
    [
        {"split": "train", "wer": trainer.predict(prepared_datasets["train"]).metrics["test_wer"]},
        {"split": "validation", "wer": trainer.predict(prepared_datasets["validate"]).metrics["test_wer"]},
    ]
)

################################
trainer.train()

################################
import pandas as pd

pd.DataFrame(
    [
        {"split": "train", "wer": trainer.predict(prepared_datasets["train"]).metrics["test_wer"]},
        {"split": "validation", "wer": trainer.predict(prepared_datasets["validate"]).metrics["test_wer"]},
    ]
)

################################
trainer.save_model("/content/drive/MyDrive/ja_voice_finetuning/models")

################################
