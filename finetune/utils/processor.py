from typing import Dict, List, Union
from dataclasses import dataclass
import evaluate
import spacy
import ginza
import torch


class Processor:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        self.metric = evaluate.load("wer")
        self.nlp = spacy.load("ja_ginza")
        ginza.set_split_mode(self.nlp, "C")  # CはNEologdの意らしいです

    def __call__(self, batch): ...

    def prepare_dataset(self, batch):
        audio = batch["audio"]

        # 音響特徴量抽出
        batch["input_features"] = self.processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # 正解のテキストをlabel idにエンコード
        batch["labels"] = self.processor.tokenizer(batch["correct"]).input_ids
        return batch

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 分かち書きして空白区切りに変換
        pred_str = [" ".join([str(i) for i in self.nlp(j)]) for j in pred_str]
        label_str = [" ".join([str(i) for i in self.nlp(j)]) for j in label_str]

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def feature_extractor(self, *args, **kwargs):
        # return self.processor.feature_extractor(audio, sampling_rate=16000)
        return self.processor.feature_extractor(*args, **kwargs)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Processor):
        self.processor = processor

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
