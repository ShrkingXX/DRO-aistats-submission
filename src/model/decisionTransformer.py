import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gpytorch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

class DecisionTransformer(nn.Module):
    def __init__(self, config, input_dim, action_dim):
        super(DecisionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length
        
        # Embeddings
        self.state_embedding = nn.Linear(input_dim, config.hidden_size)
        self.action_embedding = nn.Linear(action_dim, config.hidden_size)
        self.reward_embedding = nn.Linear(1, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, action_dim)
        )
        
    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Create embeddings for states, actions, rewards
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        reward_embeddings = self.reward_embedding(rewards.unsqueeze(-1))
        
        # Create position embeddings
        position_embeddings = self.position_embedding(timesteps)
        
        # Combine embeddings for sequence input
        sequence = torch.stack([state_embeddings, action_embeddings, reward_embeddings], dim=1)
        sequence = sequence.view(batch_size, 3 * seq_length, self.hidden_size)
        sequence = sequence + position_embeddings.repeat_interleave(3, dim=1)
        
        # Apply transformer
        if attention_mask is not None:
            # Repeat mask for state, action, reward triplets
            mask = attention_mask.repeat_interleave(3, dim=1)
            transformer_outputs = self.transformer(sequence, src_key_padding_mask=~mask)
        else:
            transformer_outputs = self.transformer(sequence)
        
        # Extract action predictions (every third output corresponds to an action)
        action_outputs = transformer_outputs[:, 1::3]
        
        # Predict next actions
        predicted_actions = self.action_head(action_outputs)
        
        return predicted_actions
