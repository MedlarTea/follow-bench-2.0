# ===== Python 标准库 =====
"""
OpenTrackVLA Mixed Model - 支持 Navigation 和 QA 两种任务的混合训练模型

Navigation 任务：
    - 输入：coarse_tokens + fine_tokens + instruction + yaw_hist + yaw_curr
    - 输出：waypoint predictions (B, n_waypoints, 3)
    - 使用 action query token + PlannerHead

QA 任务：
    - 输入：coarse_tokens + question + answer (teacher forcing)
    - 输出：next-token prediction loss
    - 使用 CausalLM 的 LM Head
"""
import json
import math
from typing import List, Optional, Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

# ===== PyTorch =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ===== HuggingFace Transformers =====
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ===== PEFT / LoRA =====
from peft import LoraConfig, get_peft_model, TaskType
from socialRPF.constants import VIEW_YAWS


def load_model_config(path):
    with open(path, "r") as f:
        data = json.load(f)
    return ModelConfig(**data)


@dataclass
class ModelConfig:
    """混合模型配置"""
    pretrained_ckpt: str = ""
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = False
    view_list: Optional[List[str]] = None
    
    # --- LoRA 配置 ---
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False
    
    # --- Navigation 配置 ---
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = True
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = 0.535
    alpha_yaw: Optional[float] = 1.572
    
    # --- QA 配置 ---
    beta_qa: float = 1.0
    max_answer_length: int = 256

    # --- 选择性 Freeze（默认都不冻结，保持原行为；QA-only warmup 阶段可以单独冻结）---
    freeze_proj: bool = False         # CrossModalityProjector
    freeze_tvi: bool = False          # TVIEmbedder（含 base_emb / yaw_proj / time_proj / bbox_proj）
    freeze_planner: bool = False      # PlannerHead3L（轨迹回归头）
    freeze_act_token: bool = False    # <act> nn.Parameter


class TVIEmbedder(nn.Module):
    """Temporal-Viewpoint Indicator with token insertion."""
    def __init__(self, d_model: int):
        super().__init__()
        self.base_emb = nn.Embedding(1, d_model)
        self.yaw_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.bbox_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def make_tvi_token(self, t_scalar: int, theta: float, device: Optional[torch.device] = None) -> torch.Tensor:
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor(
            [math.sin(theta), math.cos(theta)],
            dtype=next(self.yaw_proj.parameters()).dtype,
            device=device
        )
        yaw_embedding = self.yaw_proj(sincos)
        t_tensor = torch.tensor(
            t_scalar,
            dtype=next(self.time_proj.parameters()).dtype,
            device=device
        )
        t_embedding = self.time_proj(t_tensor.unsqueeze(0))
        tok = self.base_emb.weight[0] + yaw_embedding + t_embedding
        return tok.to(device) if device is not None else tok

    def make_ti_token(self, t_scalar: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Time-only indicator: base + time, 无 yaw/bbox（用于单视角 QA 数据）"""
        t_tensor = torch.tensor(
            t_scalar,
            dtype=next(self.time_proj.parameters()).dtype,
            device=device
        )
        t_embedding = self.time_proj(t_tensor.unsqueeze(0))
        tok = self.base_emb.weight[0] + t_embedding
        return tok.to(device) if device is not None else tok
    
    def make_tvbi_token(self, t_scalar: int, theta: float, bbox, device: Optional[torch.device] = None) -> torch.Tensor:
        bbox_dtype = next(self.bbox_proj.parameters()).dtype
        bbox_in = bbox.to(dtype=bbox_dtype, device=device) if device is not None else bbox.to(dtype=bbox_dtype)
        bbox_embedding = self.bbox_proj(bbox_in)

        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        sincos = torch.tensor(
            [math.sin(theta), math.cos(theta)],
            dtype=next(self.yaw_proj.parameters()).dtype,
            device=device
        )

        yaw_embedding = self.yaw_proj(sincos)

        t_tensor = torch.tensor(
            t_scalar,
            dtype=next(self.time_proj.parameters()).dtype,
            device=device
        )
        t_embedding = self.time_proj(t_tensor.unsqueeze(0))
        tok = self.base_emb.weight[0] + yaw_embedding + t_embedding + bbox_embedding
        return tok.to(device) if device is not None else tok


class CrossModalityProjector(nn.Module):
    """Vision to LLM space projector."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PlannerHead3L(nn.Module):
    """Three-layer MLP mapping to normalized waypoints â ∈ [-1,1]."""
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh

    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)
        return y.view(-1, self.nw, self.ad)


# ----------------------- Mixed Model -----------------------
class OpenTrackVLAMixed(nn.Module):
    """
    Multi-task model supporting both:
    1. Navigation: visual observations -> waypoint prediction (regression)
    2. QA: visual observations + question -> text answer (next-token prediction)
    
    Architecture:
        - Shared: LLM backbone, Vision Projector, TVI Embedder
        - Nav-specific: Action Query Token, Planner Head
        - QA-specific: LM Head (from CausalLM)
    """

    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.view_list = cfg.view_list or ['forward']

        if not all(v in VIEW_YAWS for v in self.view_list):
            raise ValueError(
                f"Invalid view_list: {self.view_list}. "
                f"Must be in {list(VIEW_YAWS.keys())}"
            )

        # ========== Load LLM (CausalLM for QA) ==========
        # 尝试启用 Flash Attention 2 (需要安装 flash-attn 包)
        # 如果不可用，回退到 sdpa (PyTorch 2.0+ 原生支持) 或 eager
        attn_impl = "eager"
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            # Flash Attention 未安装，尝试使用 PyTorch SDPA
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                attn_impl = "sdpa"
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[MODEL] Using attention implementation: {attn_impl}")
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=attn_impl,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm_name,
        )
        
        # 确保有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ========== Gradient Checkpointing ==========
        if cfg.gradient_checkpointing:
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                self.llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # ========== LoRA ==========
        if cfg.use_lora:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"[LoRA] Applying LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}")

            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.CAUSAL_LM,
                bias="none"
            )
            self.llm.config.use_cache = False
            self.llm = get_peft_model(self.llm, lora_config)
            if rank == 0:
                self.llm.print_trainable_parameters()
        else:
            self.llm.requires_grad_(not cfg.freeze_llm)

        # ========== Model Dimensions ==========
        self.D = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        # ========== Shared Components ==========
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.proj.requires_grad_(not cfg.freeze_proj)

        self.tvi = TVIEmbedder(self.D)
        self.tvi.requires_grad_(not cfg.freeze_tvi)

        # ========== Navigation Components ==========
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        self.act_token.requires_grad_(not cfg.freeze_act_token)

        action_dims = 3  # (x, y, yaw)
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        self.planner.requires_grad_(not cfg.freeze_planner)

        # Alpha scaling for navigation
        alpha_vec = torch.ones(1, 1, action_dims)
        if cfg.alpha_xy is not None:
            alpha_vec[0, 0, 0] = cfg.alpha_xy
            alpha_vec[0, 0, 1] = cfg.alpha_xy
        if cfg.alpha_yaw is not None:
            alpha_vec[0, 0, 2] = cfg.alpha_yaw
        self.register_buffer("alpha_task", alpha_vec)

        # ========== Trainable 参数统计（仅 rank0）==========
        if rank == 0:
            any_frozen = (cfg.freeze_llm or cfg.freeze_proj or cfg.freeze_tvi
                          or cfg.freeze_planner or cfg.freeze_act_token)
            if any_frozen:
                n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in self.parameters())
                print(f"[MODEL] Freeze config: llm={cfg.freeze_llm}, proj={cfg.freeze_proj}, "
                      f"tvi={cfg.freeze_tvi}, planner={cfg.freeze_planner}, "
                      f"act_token={cfg.freeze_act_token}")
                print(f"[MODEL] Trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M "
                      f"({100 * n_trainable / n_total:.2f}%)")

    # ==================== Helper Methods ====================
    
    # def _get_base_model(self):
    #     """获取底层的 transformer model (处理 PEFT wrapper)"""
    #     if hasattr(self.llm, 'base_model'):
    #         # PEFT wrapped
    #         return self.llm.base_model.model.model
    #     elif hasattr(self.llm, 'model'):
    #         # Standard CausalLM
    #         return self.llm.model
    #     else:
    #         return self.llm

    def _embed_text(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and embed text.
        
        Note: padding='longest' means padding to the longest sequence in the batch,
        not to max_length. max_length only serves as a truncation limit.
        """
        tok = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',  # 动态 padding 到 batch 中最长序列
            truncation=True,
            max_length=512  # 仅作为截断上限，防止异常长文本
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _embed_text_from_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed from pre-tokenized ids (e.g. from collate). No tokenizer call."""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        emb = self.llm.get_input_embeddings()(input_ids)
        return emb, attention_mask

    def _tokenize_answers(self, answers: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize answers for teacher forcing with EOS token appended."""
        tok = self.tokenizer(
            answers,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.cfg.max_answer_length - 1,  # Reserve space for EOS
            add_special_tokens=True
        )
        input_ids = tok['input_ids'].to(device)
        attention_mask = tok['attention_mask'].to(device)
        
        # Append EOS token to each sequence (so model learns to generate terminator)
        B = input_ids.size(0)
        eos_id = self.tokenizer.eos_token_id
        eos_col = torch.full((B, 1), eos_id, dtype=input_ids.dtype, device=device)
        eos_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
        
        input_ids = torch.cat([input_ids, eos_col], dim=1)
        attention_mask = torch.cat([attention_mask, eos_mask], dim=1)
        
        return input_ids, attention_mask

    def _clean_generated_text(self, text: str) -> str:
        """Remove special tokens and clean up generated text."""
        # Remove common special tokens
        special_tokens = [
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            self.tokenizer.unk_token,
        ]
        for token in special_tokens:
            if token is not None:
                text = text.replace(token, '')
        return text.strip()

    def _interleave_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        token_size: int,
        yaw_per_frame: Optional[torch.Tensor] = None,
        skip_yaw: bool = False,
    ) -> torch.Tensor:
        """
        Insert TVI tokens between visual tokens.
        
        Args:
            tokens: (B, N, D) visual tokens
            t_idx: (B, N) time indices per token
            token_size: tokens per time step (4 for coarse, 64 for fine)
            yaw_per_frame: (B, num_frames * num_views) yaw angles
            skip_yaw: True 时使用 TI-only token（base+time），
                      按单视角处理（每 token_size 个 token 为一步）
        
        Returns:
            (B, N + num_tvi_tokens, D) tokens with TVI inserted
        """
        B, N, D = tokens.shape
        out_list = []

        num_views = 1 if skip_yaw else len(self.view_list)

        if not skip_yaw and yaw_per_frame is not None:
            yaw_per_token = torch.repeat_interleave(yaw_per_frame, repeats=token_size, dim=-1)
        else:
            yaw_per_token = None

        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0

            while i < N:
                tcur = int(tb[i].item())

                for v_idx in range(num_views):
                    start = i + v_idx * token_size
                    end = start + token_size
                    if end > N:
                        break

                    if skip_yaw:
                        tok = self.tvi.make_ti_token(tcur, device=xb.device).unsqueeze(0)
                    elif yaw_per_token is not None:
                        theta = float(yaw_per_token[b, start].item())
                        tok = self.tvi.make_tvi_token(tcur, theta, device=xb.device).unsqueeze(0)
                    else:
                        theta = VIEW_YAWS.get(self.view_list[v_idx], 0.0)
                        tok = self.tvi.make_tvi_token(tcur, theta, device=xb.device).unsqueeze(0)

                    items.append(tok)
                    items.append(xb[start:end])

                i += num_views * token_size

            out_list.append(torch.cat(items, dim=0))

        return torch.stack(out_list, dim=0)

    # ==================== Navigation Forward ====================
    
    def forward_navigation(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: Optional[List[str]] = None,
        instruction_input_ids: Optional[torch.Tensor] = None,
        instruction_attention_mask: Optional[torch.Tensor] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        yaw_curr: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for navigation task.
        
        Args:
            coarse_tokens: (B, Nc, D) coarse visual tokens
            coarse_tidx: (B, Nc) time indices for coarse tokens
            fine_tokens: (B, Nf, D) fine visual tokens
            fine_tidx: (B, Nf) time indices for fine tokens
            instructions: List[str] navigation instructions (used when instruction_input_ids not provided)
            instruction_input_ids: Pre-tokenized instruction ids from collate (optional)
            instruction_attention_mask: Pre-tokenized attention mask (optional)
            yaw_hist: (B, num_hist_frames * num_views) historical yaw angles
            yaw_curr: (B, num_views) current yaw angles
            alpha: Optional per-sample scaling factor.
                   Shape (B, 3) or (B, 1, 3). When provided, overrides self.alpha_task.
                   Each row is [alpha_xy, alpha_xy, alpha_yaw].
        
        Returns:
            tau_pred: (B, n_waypoints, 3) predicted waypoints in absolute units
        """
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)

        # Project visual tokens to LLM space
        _dtype = next(self.proj.parameters()).dtype
        vis_c = self.proj(coarse_tokens.to(device=device, dtype=_dtype))
        vis_f = self.proj(fine_tokens.to(device=device, dtype=_dtype))

        # print("[DEBUG] vis_c shape:", vis_c.shape)
        # print("[DEBUG] vis_f shape:", vis_f.shape)

        # Insert TVI tokens
        vis_c = self._interleave_tvi(vis_c, coarse_tidx.to(device), token_size=4, yaw_per_frame=yaw_hist)
        vis_f = self._interleave_tvi(vis_f, fine_tidx.to(device), token_size=64, yaw_per_frame=yaw_curr)

        # print("[DEBUG] vis_c shape after TVI:", vis_c.shape)

        # Embed text instructions (pre-tokenized in collate or tokenize here)
        if instruction_input_ids is not None and instruction_attention_mask is not None:
            txt_emb, txt_mask = self._embed_text_from_ids(
                instruction_input_ids, instruction_attention_mask, device
            )
        else:
            if instructions is None:
                raise ValueError("Either instructions or (instruction_input_ids, instruction_attention_mask) must be provided")
            txt_emb, txt_mask = self._embed_text(instructions, device)

        # print("[DEBUG] txt_emb shape:", txt_emb.shape)
        # print("[DEBUG] txt_mask shape:", txt_mask.shape)

        # Action query token
        act = self.act_token.expand(B, 1, -1)

        # Concatenate: [text, vis_coarse, vis_fine, action_query]
        seq = torch.cat([txt_emb, vis_c, vis_f, act], dim=1).to(self.llm.dtype)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)
        ], dim=1)

        # Forward through LLM backbone
        # base_model = self._get_base_model()
        # out = base_model(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)

        # Extract action hidden state (last position)
        # h_act = out.last_hidden_state[:, -1, :]
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.hidden_states[-1][:, -1, :]
        h_act = h_act.to(next(self.planner.parameters()).dtype)

        # Predict waypoints
        a_hat = self.planner(h_act)  # normalized [-1, 1]
        if alpha is not None:
            # Per-sample alpha: (B, 3) → (B, 1, 3) for broadcasting with (B, n_waypoints, 3)
            if alpha.dim() == 2:
                alpha = alpha.unsqueeze(1)
            tau_pred = a_hat * alpha.to(a_hat.device, a_hat.dtype)
        else:
            tau_pred = a_hat * self.alpha_task  # fallback to global alpha

        return tau_pred.float()

    # ==================== QA Forward ====================
    
    def forward_qa(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        instructions: Optional[List[str]] = None,
        instruction_input_ids: Optional[torch.Tensor] = None,
        instruction_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[List[str]] = None,
        yaw_hist: Optional[torch.Tensor] = None,
        return_pred_text: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[List[str]]]:
        """
        Forward pass for QA task with teacher forcing.
        
        Args:
            coarse_tokens: (B, N, D) visual tokens (静态图像复制为视频)
            coarse_tidx: (B, N) time indices
            instructions: List[str] questions (used when instruction_input_ids not provided)
            instruction_input_ids: Pre-tokenized question ids from collate (optional)
            instruction_attention_mask: Pre-tokenized attention mask (optional)
            labels: List[str] answers for training, None for inference
            yaw_hist: Optional yaw angles
            return_pred_text: If True, return predicted text for debugging
        
        Returns:
            If labels provided: (loss, logits, pred_texts) where pred_texts is None if return_pred_text=False
            Else: (None, logits, None)
        """
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)

        # Project visual tokens
        _dtype = next(self.proj.parameters()).dtype
        vis_c = self.proj(coarse_tokens.to(device=device, dtype=_dtype))
        vis_c = self._interleave_tvi(vis_c, coarse_tidx.to(device), token_size=4, skip_yaw=True)

        # Embed questions (pre-tokenized in collate or tokenize here)
        if instruction_input_ids is not None and instruction_attention_mask is not None:
            txt_emb, txt_mask = self._embed_text_from_ids(
                instruction_input_ids, instruction_attention_mask, device
            )
        else:
            if instructions is None:
                raise ValueError("Either instructions or (instruction_input_ids, instruction_attention_mask) must be provided")
            txt_emb, txt_mask = self._embed_text(instructions, device)
        txt_len = txt_emb.size(1)
        vis_len = vis_c.size(1)

        if labels is not None:
            # ========== Training Mode (Teacher Forcing) ==========
            # Tokenize answers
            answer_ids, answer_mask = self._tokenize_answers(labels, device)
            answer_emb = self.llm.get_input_embeddings()(answer_ids)
            answer_len = answer_ids.size(1)

            # Concatenate: [question, vision, answer]
            seq = torch.cat([txt_emb, vis_c, answer_emb], dim=1).to(self.llm.dtype)
            attn = torch.cat([
                txt_mask,
                torch.ones(B, vis_len, dtype=torch.long, device=device),
                answer_mask
            ], dim=1)

            # Forward through CausalLM (包含 LM Head)
            outputs = self.llm(
                inputs_embeds=seq,
                attention_mask=attn,
                use_cache=False
            )
            logits = outputs.logits  # (B, L, vocab_size)

            # Compute loss: only on answer tokens (shifted)
            # logits 在 [txt + vis + answer] 上
            # 我们需要预测 answer 的下一个 token
            # 即 logits[:, txt_len + vis_len - 1 : txt_len + vis_len + answer_len - 1] 预测 answer_ids[:, 1:]
            
            start_pos = txt_len + vis_len - 1
            end_pos = start_pos + answer_len
            shift_logits = logits[:, start_pos:end_pos, :].float().contiguous()
            
            shift_labels = answer_ids.contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Decode predicted text for debugging
            pred_texts = None
            if return_pred_text:
                pred_ids = shift_logits.argmax(dim=-1)  # (B, answer_len)
                pred_texts = []
                for i in range(B):
                    # Mask out padding positions
                    valid_mask = answer_mask[i] == 1
                    valid_ids = pred_ids[i][valid_mask[:pred_ids.size(1)]]
                    text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                    pred_texts.append(self._clean_generated_text(text))

            return loss, logits, pred_texts
        else:
            # ========== Inference Mode ==========
            seq = torch.cat([txt_emb, vis_c], dim=1).to(self.llm.dtype)
            attn = torch.cat([
                txt_mask,
                torch.ones(B, vis_len, dtype=torch.long, device=device)
            ], dim=1)

            outputs = self.llm(
                inputs_embeds=seq,
                attention_mask=attn,
                use_cache=False
            )
            logits = outputs.logits

            return None, logits, None

    # ==================== Unified Forward ====================
    
    def forward(
        self,
        task_type: Literal['nav', 'qa'],
        **kwargs
    ):
        """
        Unified forward interface.
        
        Args:
            task_type: 'nav' for navigation, 'qa' for question answering
            **kwargs: task-specific arguments
        
        Returns:
            For 'nav': tau_pred (B, n_waypoints, 3)
            For 'qa': (loss, logits) or (None, logits)
        """
        if task_type == 'nav':
            return self.forward_navigation(**kwargs)
        elif task_type == 'qa':
            return self.forward_qa(**kwargs)
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Must be 'nav' or 'qa'")

    # ==================== Generation (Inference) ====================
    
    @torch.inference_mode()
    def generate_answer(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        question: str,
        max_length: int = 128,
        yaw_hist: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text answer for QA task (autoregressive) with KV Cache acceleration.
        
        Args:
            coarse_tokens: (N, D) visual tokens for single sample
            coarse_tidx: (N,) time indices
            question: str question
            max_length: maximum generation length
            yaw_hist: Optional yaw angles
            temperature: sampling temperature
            top_p: nucleus sampling threshold
        
        Returns:
            Generated answer string
        """
        device = next(self.parameters()).device

        # Add batch dimension if needed
        if coarse_tokens.dim() == 2:
            coarse_tokens = coarse_tokens.unsqueeze(0)
            coarse_tidx = coarse_tidx.unsqueeze(0)
            if yaw_hist is not None:
                yaw_hist = yaw_hist.unsqueeze(0)

        # Encode visual + text
        _dtype = next(self.proj.parameters()).dtype
        vis_c = self.proj(coarse_tokens.to(device=device, dtype=_dtype))
        vis_c = self._interleave_tvi(vis_c, coarse_tidx.to(device), token_size=4, yaw_per_frame=yaw_hist)

        txt_emb, txt_mask = self._embed_text([question], device)

        # Initial sequence: [question, vision]
        seq = torch.cat([txt_emb, vis_c], dim=1).to(self.llm.dtype)
        attn = torch.cat([
            txt_mask,
            torch.ones(1, vis_c.size(1), dtype=torch.long, device=device)
        ], dim=1)

        # First forward pass: encode context and get KV Cache
        outputs = self.llm(inputs_embeds=seq, attention_mask=attn, use_cache=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :] / temperature

        # Generate tokens autoregressively with KV Cache
        generated_ids = []
        for _ in range(max_length):
            # Nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id.item())

            # Get embedding for next token only
            next_emb = self.llm.get_input_embeddings()(next_token_id).to(self.llm.dtype)
            
            # Update attention mask (append 1 for new token)
            attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
            
            # Forward only the new token with KV Cache (avoid recomputing entire sequence)
            outputs = self.llm(
                inputs_embeds=next_emb,
                attention_mask=attn,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :] / temperature

        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = self._clean_generated_text(answer)
        return answer
