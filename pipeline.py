from transformers import AutoTokenizer, AutoModelForCausalLM


repo_id = "rinna/japanese-gpt2-xsmall"


class KigosHaikuPipeline:
    def __init__(self, model_path_or_url: str | None = None, tokenizer_path_or_url: str | None = None, use_fast: bool = False, do_lower_case: bool = True) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_path_or_url or repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_url or repo_id, use_fast=use_fast)
        self.tokenizer.do_lowercase = do_lower_case

    
    def encapsulate_content(self, kigos: str | list[str], haiku: str | None = None) -> str:
        # format: '<s>Kigos[SEP]Haiku</s>' + [PAD] until reaching max length
        kigo_list = ",".join(kigos) if isinstance(kigos, list) else kigos
        sentence = self.tokenizer.bos_token + kigo_list + self.tokenizer.sep_token
        if haiku is not None:
            sentence += haiku + self.tokenizer.eos_token
        return sentence


    def sample_haikus(self, kigos: str | list[str], n_samples: int, max_length: int = 40) -> list[str]:
        sentence = self.encapsulate_content(kigos)
        input = self.tokenizer.encode(sentence, return_tensors="pt")
        outputs = self.model.generate(input, do_sample=True, num_return_sequences=n_samples, max_length=max_length)
        results = []
        for output_ids in outputs:
            output_sentence = self.tokenizer.decode(output_ids.tolist())
            results.append(output_sentence)
        return results
