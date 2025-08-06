import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from model import layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x):
        print(f"Input Shape: {x.shape}")
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
            x = x * 1.5
        
        return x

class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x


class STEPmr(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STEPmr, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32) 
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockTwoSTBlocks(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        self.connectivity_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]//2),
            nn.LayerNorm(blocks[-1][0]//2 ),
            nn.ELU(),
            nn.Linear(blocks[-1][0]//2, 1),
            nn.Sigmoid()
        )
        
        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        identity = x
        
        x = self.st_blocks(x)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        
        learned_attention = self.connectivity_attention(x)  # [batch, time_steps, nodes, 1]
      
        connectivity_weights = self.connection_weights
        connectivity_weights = connectivity_weights.view(1, 1, -1, 1)
        
        attention = (learned_attention * (1.0 + self.attention_scale * connectivity_weights))
        attention = F.layer_norm(attention, [attention.size(-1)])

        x = x * attention + 0.1 * x  
        
        x = self.expression_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return x


class STEPmi(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STEPmi, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(connections, dim=0)
        print(f"Connection weights: {self.connection_weights}")
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockTwoSTBlocks(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        #self.global_attention_scale = nn.Parameter(torch.tensor(0.4))
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        print(f"Ko: {self.Ko}")

        self.lstm_mirna = nn.LSTM(
            input_size=blocks[-3][-1], 
            hidden_size=blocks[-3][-1],
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        print(f"Blocks [-3][-1] --> input for LSTM: {blocks[-3][-1]}") # 192

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}") # 192
    
        self.lstm_proj = nn.Linear(2 * blocks[-3][-1], blocks[-3][-1])  # *2 for bidirectional
        
        self.lstm_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])

        print(f"Input for MultiHeadAttention --> blocks[-1][0]: {blocks[-1][0]}") # 224
 
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=blocks[-1][0],  # Feature dimension after output block
            num_heads=4,
            dropout=0.3
        )

        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            print(f"Model process layers.OutputBlock!")
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            print(f"Model process Ko==0 part!")
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

        self.expression_proj_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    
        self._init_weights()

        
    def forward(self, x):
        x = self.st_blocks(x)
        #print(f"Shape after ST blocks: {x.shape}") 

        batch_size, features, time_steps, nodes = x.shape
        #print(f"Batch size: {batch_size}, Features: {features}, Time steps: {time_steps}, Nodes: {nodes}")

        # LSTM processing
        x_lstm = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        #print(f"Shape after permute for LSTM: {x_lstm.shape}")

        x_lstm = x_lstm.reshape(batch_size * nodes, time_steps, features)
        #print(f"Shape before LSTM: {x_lstm.shape}")
        
        lstm_out, _ = self.lstm_mirna(x_lstm)
        #print(f"Shape after LSTM: {lstm_out.shape}")

        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, nodes, time_steps, features)
        #print(f"Shape after LSTM projection: {lstm_out.shape}")
        lstm_out = lstm_out.permute(0, 3, 2, 1)  # [batch, features, time_steps, nodes]
        #print(f"Shape after permute back: {lstm_out.shape}")
        
        # Residual connection  
        x = x + lstm_out
        x = self.lstm_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Output block processing
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        #print(f"Shape before attention: {x.shape}")
        
        # Get current dimensions after output processing
        _, current_features, time_steps, nodes = x.shape
        
        x_attention = x.permute(2, 0, 3, 1)  # [time_steps, batch, nodes, features]
        x_attention = x_attention.reshape(time_steps, batch_size * nodes, current_features)
        
        attn_output, _ = self.multihead_attention(x_attention, x_attention, x_attention)

        attn_output = attn_output.reshape(time_steps, batch_size, nodes, current_features)
        attn_output = attn_output.permute(1, 3, 0, 2)  # [batch, features, time_steps, nodes]
        
        x = x + 0.4 * attn_output
        
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_mirna(x)
        #print(f"Shape after expression projection: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)