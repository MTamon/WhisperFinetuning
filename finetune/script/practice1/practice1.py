import sys
from tqdm import tqdm
from datasets import DatasetDict
from transformers import AutoProcessor
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import AutoModelForSpeechSeq2Seq


from finetune.utils import Processor, DataCollatorSpeechSeq2SeqWithPadding, make_dataset_from_path


train_dataset, valid_dataset = make_dataset_from_path("data/...", train_ratio=0.7)
datasets = DatasetDict({"train": train_dataset, "valid": valid_dataset})

whisper_processor = AutoProcessor.from_pretrained("kotoba-tech/kotoba-whisper-v1.0")
processor = Processor(whisper_processor)
prepared_datasets = datasets.map(processor.prepare_dataset, remove_columns=datasets.column_names["train"])

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

model = AutoModelForSpeechSeq2Seq.from_pretrained("kotoba-tech/kotoba-whisper-v1.0")
if model.generator.config.language != "ja":
    res = input("This model is not trained on Japanese data. Continue? (y/n)")
    if res.lower() != "n":
        sys.exit(0)
model.generation_config.language = "ja"

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=prepared_datasets["train"],
    eval_dataset=prepared_datasets["valid"],
    data_collator=data_collator,
    compute_metrics=processor.compute_metrics,
    tokenizer=processor.feature_extractor,
)

with open("out/kotoba-whisper-v1.0/pre_metrics.json", mode="w", encoding="utf-8") as f:
    train_wer = trainer.predict(prepared_datasets["train"]).metrics["test_wer"]
    valid_wer = trainer.predict(prepared_datasets["valid"]).metrics["test_wer"]
    f.write(f"train_wer: {train_wer}\nvalid_wer: {valid_wer}")
# valid の推論結果を個別に保存
with open("out/kotoba-whisper-v1.0/pre_valid_inference.csv", mode="w", encoding="utf-8") as f:
    f.write("No, pred, ref\n")
    desc = "before fine-tuning verification"
    for i, pred in enumerate(tqdm(trainer.predict(prepared_datasets["valid"]).predictions[:200]), desc=desc):
        ref = prepared_datasets["valid"][i]["correct"]
        f.write(f"{i}, {pred['pred']}, {ref}\n")

################################
trainer.train()

################################

with open("out/kotoba-whisper-v1.0/post_metrics.json", mode="w", encoding="utf-8") as f:
    train_wer = trainer.predict(prepared_datasets["train"]).metrics["test_wer"]
    valid_wer = trainer.predict(prepared_datasets["valid"]).metrics["test_wer"]
    f.write(f"train_wer: {train_wer}\nvalid_wer: {valid_wer}")
# valid の推論結果を個別に保存
with open("out/kotoba-whisper-v1.0/post_valid_inference.csv", mode="w", encoding="utf-8") as f:
    f.write("No, pred, ref\n")
    desc = "after fine-tuning verification"
    for i, pred in enumerate(tqdm(trainer.predict(prepared_datasets["valid"]).predictions[:200]), desc=desc):
        ref = prepared_datasets["valid"][i]["correct"]
        f.write(f"{i}, {pred['pred']}, {ref}\n")

################################
trainer.save_model("models")

################################
