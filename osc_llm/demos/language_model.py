import lightning as L
from pathlib import Path
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import requests
from torch.utils.data import Dataset, DataLoader
from typing import Any, Tuple
from torch import Tensor
import torch
from ..optimizers import get_cosine_lr_scheduler
from ..utils import build_model


class TangShi(Dataset):
    def __init__(self, 
                 data_dir: str = 'data',
                 block_size: int = 40,
                 download: bool = True,
                 ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "tangshi.txt"
        self.block_size = block_size
        if download:
            self.download()
        self.data, self.vocab = self.tokenize()
        
    def __len__(self) -> int:
        return len(self.data) // self.block_size
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start = index * self.block_size
        end = start + self.block_size
        inputs = self.data[start:end]
        target = self.data[(start + 1) : (end + 1)]
        return inputs, target
    
    def download(self):
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        if self.path.exists():
            return
        url = 'https://deepasset.oss-cn-beijing.aliyuncs.com/tangshi.txt'
        with open(self.path, 'x') as f:
            f.write(requests.get(url).text)
            
    def tokenize(self):
        ids = []
        vocab = Vocab()
        vocab.add_token('<pad>')
        vocab.add_token('<s>')
        vocab.add_token('</s>')
        with open(self.path, encoding="utf8") as f:
            for line in f:
                chars = list(line.strip())
                for char in chars:
                    vocab.add_token(char)
                    ids.append(vocab.token2id[char])
                ids.append(vocab.token2id['</s>'])
        return torch.tensor(ids, dtype=torch.long), vocab
        
                  
class Vocab():
    def __init__(self) -> None:
        self.token2id = {}
        self.id2token = {}
    
    @property
    def vocab_size(self):
        return len(self.token2id)
    
    def __len__(self):
        return len(self.token2id)
        
    def add_token(self, token):
        if token not in self.token2id:
            self.id2token[len(self.token2id)] = token
            self.token2id[token] = len(self.token2id)
            
            
            
class TangshiLanguageModel(L.LightningModule):
    def __init__(self,
                 config: str,
                 lr: float = 1e-4,
                 block_size: int = 100,
                 batch_size: int = 16) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(config=config, empty_init=False)
        self.dataset = TangShi(block_size=block_size)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
        
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warm_up_steps = min(100, num_training_steps // 10)
        scheduler = get_cosine_lr_scheduler(optimizer, 
                                            num_training_steps=num_training_steps, 
                                            num_warm_up_steps=num_warm_up_steps,
                                            min_lr_ratio=0.01)
        scheduler_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler_config]
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        inputs, target = batch
        logits = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss
    
    def train_dataloader(self) -> Any:
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def generate(self, 
                 prompt: str, 
                 top_k: int = 5, 
                 max_len: int = 100, 
                 temperature: float = 1.0):
        ids = [self.dataset.vocab.token2id[char] for char in prompt]
        input_pos = torch.arange(len(ids), dtype=torch.long, device=self.device)
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.model.eval()
        self.model.build_kv_caches(batch_size=1, device=self.device)
        outputs = []
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.model(ids.reshape(1, -1), input_pos)
                logits = logits[0, -1] / temperature
                # 选择概率最大的top_k个token, 其他的token logits设置为负无穷
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                ids = torch.multinomial(probs, num_samples=1)
                if ids[0] == self.dataset.vocab.token2id['</s>']:
                    break
                input_pos = input_pos[-1:] + 1
                outputs.append(self.dataset.vocab.id2token[ids[0].item()])
        return prompt + "".join(outputs)           