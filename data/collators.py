from transformers import DataCollatorForSeq2Seq

class IdDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        ids = [f.pop("idxs") for f in features]

        #superclass collator only accept specific featues
        batch = super().__call__(features)
        batch["ids"] = ids
        return batch