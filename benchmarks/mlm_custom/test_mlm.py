import pandas as pd
import numpy as np
from transformers import AutoTokenizer, RobertaModel, AutoModel, AutoModelForMaskedLM
from transformers import pipeline
import os
import json


class MLMTest():

    def __init__(self, config_file="mlm_test_config.csv", full_text_file="mlm_full_text.csv", targeted_text_file="mlm_targeted_text.csv"):

        self.config_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file))
        self.config_df.fillna("", inplace=True)
        self.full_text_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), full_text_file))
        self.targeted_text_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), targeted_text_file))
        self.full_text_results = []
        self.targeted_text_results = []
        
    def _run_full_test_row(self, text, print_debug=False):
        return_data = []
        data = text.split()
        for i in range(0, len(data)):
            masked_text = " ".join(data[:i]) + " "+self.nlp.tokenizer.mask_token+" " + " ".join(data[i+1:])
            expected_result = data[i]
            result = self.nlp(masked_text)
            self.full_text_results.append({"text": masked_text, "result": result[0]["token_str"], "true_output": expected_result})
            if print_debug:
                print(masked_text)
                print([x["token_str"] for x in result])
                print("-"*20)
            return_data.append({"prediction": result[0]["token_str"], "true_output": expected_result})
        return return_data

    def _run_targeted_test_row(self, text, expected_result, print_debug=False):
        return_data = []
        result = self.nlp(text.replace("<mask>", self.nlp.tokenizer.mask_token))
        self.targeted_text_results.append({"text": text, "result": result[0]["token_str"], "true_output": expected_result})
        if print_debug:
            print(text)
            print([x["token_str"] for x in result])
            print("-"*20)
        return_data.append({"prediction": result[0]["token_str"], "true_output": expected_result})
        return return_data

    def _compute_acc(self, results):
        ctr = 0
        for row in results:
            try:
                z = json.loads(row["true_output"])
                if isinstance(z, list):
                    if row["prediction"] in z:
                        ctr+=1
            except:
                if row["prediction"] == row["true_output"]:
                    ctr+=1

        return float(ctr/len(results))

    def run_full_test(self, exclude_user_ids=[], print_debug=False):
        df = pd.DataFrame()
        for idx, row in self.config_df.iterrows():
            self.full_text_results = []
            
            model_name = row["model_name"]
            display_name = row["display_name"] if row["display_name"] else row["model_name"]
            revision = row["revision"] if row["revision"] else "main"
            from_flax = row["from_flax"]
            if from_flax:
                model = AutoModelForMaskedLM.from_pretrained(model_name, from_flax=True, revision=revision)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained('exported_pytorch_model')
                model.save_pretrained('exported_pytorch_model')
                self.nlp = pipeline('fill-mask', model="exported_pytorch_model")
            else:
                self.nlp = pipeline('fill-mask', model=model_name)
            accs = []
            try:
                for idx, row in self.full_text_df.iterrows():
                    if row["user_id"] in exclude_user_ids:
                        continue
                    results = self._run_full_test_row(row["text"], print_debug=print_debug)

                    acc = self._compute_acc(results)
                    accs.append(acc)
            except:
                print("Error for", display_name)
                continue

            print(display_name, " Average acc:", sum(accs)/len(accs))
            if df.empty:
                df = pd.DataFrame(self.full_text_results)
                df.rename(columns={"result": display_name}, inplace=True)
            else:
                preds = [x["result"] for x in self.full_text_results]
                df[display_name] = preds
        df.to_csv("full_text_results.csv", index=False)
        print("Results saved to full_text_results.csv")

    def run_targeted_test(self, exclude_user_ids=[], print_debug=False):

        df = pd.DataFrame()
        for idx, row in self.config_df.iterrows():
            self.targeted_text_results = []
            
            model_name = row["model_name"]
            display_name = row["display_name"] if row["display_name"] else row["model_name"]
            revision = row["revision"] if row["revision"] else "main"
            from_flax = row["from_flax"]
            if from_flax:
                model = AutoModelForMaskedLM.from_pretrained(model_name, from_flax=True, revision=revision)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.save_pretrained('exported_pytorch_model')
                model.save_pretrained('exported_pytorch_model')
                self.nlp = pipeline('fill-mask', model="exported_pytorch_model")
            else:
                self.nlp = pipeline('fill-mask', model=model_name)
            accs = []
            try:
                for idx, row2 in self.targeted_text_df.iterrows():
                    if row2["user_id"] in exclude_user_ids:
                        continue
                    results = self._run_targeted_test_row(row2["text"], row2["output"], print_debug=print_debug)

                    acc = self._compute_acc(results)
                    accs.append(acc)
            except:
                import traceback
                print(traceback.format_exc())
                print("Error for", display_name)
                continue

            print(display_name, " Average acc:", sum(accs)/len(accs))
            if df.empty:
                df = pd.DataFrame(self.targeted_text_results)
                df.rename(columns={"result": display_name}, inplace=True)
            else:
                preds = [x["result"] for x in self.targeted_text_results]
                df[display_name] = preds
        df.to_csv("targeted_text_results.csv", index=False)
        print("Results saved to targeted_text_results.csv")
 
