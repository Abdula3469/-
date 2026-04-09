import json
import torch
import re
import time
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    """Централизованная конфигурация"""
    model_name = "microsoft/phi-2"
    output_dir = "./phi2-sparql-con"
    final_dir = "./phi2-sparql-con-final"
    use_4bit = True
    lora_r = 128
    lora_alpha = 256
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    num_epochs = 5
    batch_size = 1
    gradient_accumulation = 8
    learning_rate = 2e-4
    warmup_steps = 0.1
    weight_decay = 0.01
    max_grad_norm = 0.5
    max_length = 512
    data_path = "training_data.jsonl"
    device_map = "auto"
    fp16 = True


class SPARQLPostProcessor:
    ENTITY_CORRECTIONS = {
        "вторая мировая война": "Q362",
        "второй мировой войны": "Q362",
        "вторую мировую войну": "Q362",
        "первая мировая война": "Q361",
        "россия": "Q159",
        "российской федерации": "Q159",
        "ссср": "Q15180",
        "советский союз": "Q15180",
        "германия": "Q183",
        "франция": "Q142",
        "италия": "Q38",
    }
    
    PROPERTY_CORRECTIONS = {
        "началась": "P580",
        "начало": "P580",
        "закончилась": "P582",
        "конец": "P582",
        "столица": "P36",
        "автор": "P50",
        "написал": "P50",
        "родился": "P19",
        "место рождения": "P19",
    }
    
    @classmethod
    def fix_sparql(cls, sparql: str, question: str = "") -> str:
        if not sparql or not sparql.strip():
            return sparql
        sparql = cls._fix_entities(sparql, question)
        sparql = cls._fix_properties(sparql, question)
        sparql = cls._fix_syntax(sparql)
        sparql = cls._fix_missing_components(sparql)
        sparql = cls._fix_duplicates(sparql)
        
        return sparql
    
    @classmethod
    def _fix_entities(cls, sparql: str, question: str) -> str:
        for pattern, correct_id in cls.ENTITY_CORRECTIONS.items():
            if pattern in question.lower():
                sparql = re.sub(r'wd:Q\d+', f'wd:{correct_id}', sparql)
                break
        
        return sparql
    
    @classmethod
    def _fix_properties(cls, sparql: str, question: str) -> str:
        for pattern, correct_prop in cls.PROPERTY_CORRECTIONS.items():
            if pattern in question.lower():
                sparql = re.sub(r'wdt:P\d+', f'wdt:{correct_prop}', sparql)
                break
        
        return sparql
    
    @classmethod
    def _fix_syntax(cls, sparql: str) -> str:
        sparql = sparql.replace('\\}', '}').replace('\\{', '{')
        sparql = sparql.replace('"""', '"')
        lines = sparql.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            if 'wdt:P' in line and not line.rstrip().endswith('.'):
                if i < len(lines) - 1 and '}' not in lines[i+1]:
                    line = line.rstrip() + ' .'
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    @classmethod
    def _fix_missing_components(cls, sparql: str) -> str:
        if "SERVICE wikibase:label" not in sparql:
            if "}" in sparql:
                service = '  SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }\n'
                last_brace = sparql.rfind('}')
                if last_brace != -1:
                    sparql = sparql[:last_brace] + '\n' + service + sparql[last_brace:]
        if "LIMIT" not in sparql.upper():
            sparql += "\nLIMIT 10"
        
        return sparql
    
    @classmethod
    def _fix_duplicates(cls, sparql: str) -> str:
        if sparql.count("SELECT") > 1:
            first_select_end = sparql.find("}", sparql.find("SELECT"))
            if first_select_end != -1:
                sparql = sparql[:first_select_end + 1]
        
        return sparql

class SPARQLValidator:
    REQUIRED_PATTERNS = [
        (r'PREFIX\s+wd:', "Отсутствует префикс wd:"),
        (r'PREFIX\s+wdt:', "Отсутствует префикс wdt:"),
        (r'SELECT\s+\?answer', "Нет SELECT ?answer"),
        (r'WHERE\s*\{', "Нет WHERE {"),
    ]
    
    @classmethod
    def validate(cls, sparql: str) -> tuple[bool, str]:
        if not sparql or not sparql.strip():
            return False, "Пустой запрос"
        
        for pattern, message in cls.REQUIRED_PATTERNS:
            if not re.search(pattern, sparql, re.IGNORECASE):
                return False, message
        
        if sparql.count('{') != sparql.count('}'):
            return False, "Непарные фигурные скобки"
        
        if 'wdt:P' not in sparql:
            return False, "Нет свойств (wdt:P...)"
        
        if 'wd:Q' not in sparql:
            return False, "Нет entity (wd:Q...)"
        
        return True, "Валидный SPARQL"

class SparqlTrainer:
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        self._setup_quantization()
        self._load_model()
        self._setup_lora()
    
    def _setup_quantization(self):
        if self.config.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            self.bnb_config = None
    
    def _load_model(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,          
                quantization_config=self.bnb_config,
                device_map=self.config.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,        
                device_map=self.config.device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
    
   
    def _setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="lora_only",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        self.model.print_trainable_parameters()
    
    def _load_and_format_data(self, file_path: str):
        formatted_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    user_msg = data['messages'][0]['content']
                    assistant_msg = data['messages'][1]['content']
                    
                    assistant_msg = SPARQLPostProcessor.fix_sparql(
                        assistant_msg, user_msg
                    )
                    
                    full_text = f"### User:\n{user_msg}\n\n### Assistant:\n{assistant_msg}"
                    formatted_data.append({"text": full_text})
                    
                except json.JSONDecodeError as e:
                    print(f"ошибка тут {line_num}: {e}")
                    continue
        
        return formatted_data
    
    def _tokenize_function(self, examples):
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def train(self):
        
        train_data = self._load_and_format_data(self.config.data_path)
        dataset = Dataset.from_list(train_data)
        
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            fp16=self.config.fp16,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            weight_decay=self.config.weight_decay,
            dataloader_pin_memory=False,
            report_to="none",
            max_grad_norm=self.config.max_grad_norm,
            load_best_model_at_end=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        effective_batch = self.config.batch_size * self.config.gradient_accumulation
        steps_per_epoch = len(tokenized_dataset) // effective_batch
        total_steps = steps_per_epoch * self.config.num_epochs
        
        start_time = time.time()
        self.trainer.train()
        training_time = time.time() - start_time
        
        self.model.save_pretrained(self.config.final_dir)
        self.tokenizer.save_pretrained(self.config.final_dir)
        
        print(f"сохранено в {self.config.final_dir}")
        
        return training_time

if __name__ == "__main__":
    
    config = Config()
    trainer = SparqlTrainer(config)
    trainer.train()