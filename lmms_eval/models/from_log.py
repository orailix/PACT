import json
import os
import re
from datetime import datetime
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("from_log")
class FromLog(lmms):
    def __init__(self, logs: str = "logs", **kwargs) -> None:
        super().__init__()
        self.logs = {}
        log_folders = logs.split(",")

        for log_folder in log_folders:
            for root, dirs, files in os.walk(log_folder):
                for file in files:
                    if file.endswith(".jsonl"):
                        # Extract the dataset name from the filename
                        dataset_name = file[file.find("_samples_") + 9 : file.find(".jsonl")]
                       
                        # Process .jsonl file
                        log_file = os.path.join(root, file)
                        print(f"Processing file: {log_file}")
                        logs = {}
                        with open(log_file, "r") as f:
                            for line in f:
                                log_data = json.loads(line.strip())
                                doc_id = log_data["doc_id"]
                                response = log_data["resps"][0]
                                logs[doc_id] = response

                        # Infer task from the dataset name or other criteria
                        task = dataset_name  # Or any other mapping logic

                        # Find the log time from filename or mark as unknown
                        pattern = re.compile(r"\d{4}_\d{4}")
                        log_time = pattern.search(file) and pattern.findall(file)[-1] or "unknown"

                        # Update the logs dictionary only if task is not processed or time is newer
                        if task not in self.logs or (
                            self.logs[task]["time"] == "unknown" or
                            datetime.strptime(log_time, "%m%d_%H%M") > datetime.strptime(self.logs[task]["time"], "%m%d_%H%M")):
                            self.logs[task] = {"time": log_time, "logs": logs}

        # Accelerator setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], \
                   "Unsupported distributed type. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self._rank != 0), desc="Model Responding")
        print(self.logs)
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            response = self.logs[task]["logs"].get(doc_id, [""])
            res.append(response[0])
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not supported")

    def generate_until_multi_round(self, requests) -> List[str]:
        return self.generate_until(requests)
