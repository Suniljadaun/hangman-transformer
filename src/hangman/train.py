"""Training loop for HangmanNet."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import HangmanNet, ModelConfig, losses


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.03
    word_loss_weight: float = 0.5
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 200
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)


def train(batcher, cfg: TrainConfig, device: str = "cuda", model: HangmanNet | None = None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = (model or HangmanNet(cfg.model)).to(device)
    loader = batcher

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )
    total_steps = max(len(loader) * cfg.epochs, 1)
    warmup = max(int(total_steps * cfg.warmup_frac), 1)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(
                    batch["tokens"],
                    batch["guessed"],
                    batch["absent"],
                    batch["lives"],
                    batch["length"],
                )
                loss_a, loss_b = losses(
                    out, batch["labels"], batch["hidden"], batch["target_present"]
                )
                loss = loss_a + cfg.word_loss_weight * loss_b

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            schedule.step()

            running += float(loss.detach())
            step += 1
            if step % cfg.log_every == 0:
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {running / cfg.log_every:.4f} "
                    f"mlm {float(loss_a):.4f} word {float(loss_b):.4f} "
                    f"lr {schedule.get_last_lr()[0]:.2e} "
                    f"[{time.time() - start:.0f}s]",
                    flush=True,
                )
                running = 0.0
    return model
