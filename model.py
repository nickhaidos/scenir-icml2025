
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.models import ARGVA
from torch_geometric.nn.pool.glob import global_add_pool
from torch_geometric.utils import (
    add_self_loops,
    batched_negative_sampling,
    remove_self_loops,
)


def global_add_aggregation(z, edge_index, batch=None):
    return global_add_pool(z, batch)


def get_model_outputs(model, dataset, device):
    latent_embeddings = []

    model.to(device)
    
    for graph in dataset:
        graph.to(device)
        latent_embeddings.append(model.latent_graph_embedding(graph.x.float(), graph.edge_index))

    latent_embeddings = torch.cat(latent_embeddings).detach().cpu().numpy()

    return latent_embeddings


class multiGINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers=1):
        super(multiGINEncoder, self).__init__()
        self.layers = layers
        if self.layers == 0:
            raise Exception("Can't build GNN Module with zero layers.")
        
        self.gnn_mu = nn.ModuleList()
        self.gnn_logvar = nn.ModuleList()
        
        if self.layers == 1:
            self.gnn_mu.append( GINConv( nn.Linear(in_channels, out_channels)  ) )
            self.gnn_logvar.append( GINConv( nn.Linear(in_channels, out_channels)  ) )
            return None
        
        for i in range(self.layers):
            if i==0:
                self.gnn_mu.append( GINConv( nn.Linear(in_channels, hidden_channels)  ) )
                self.gnn_logvar.append( GINConv( nn.Linear(in_channels, hidden_channels)  ) )
                continue
            elif i==self.layers-1:
                self.gnn_mu.append( GINConv( nn.Linear(hidden_channels, out_channels)  ) )
                self.gnn_logvar.append( GINConv( nn.Linear(hidden_channels, out_channels)  ) )
                continue
            else:
                self.gnn_mu.append( GINConv( nn.Linear(hidden_channels, hidden_channels)  ) )
                self.gnn_logvar.append( GINConv( nn.Linear(hidden_channels, hidden_channels)  ) )
                continue
            
        return None

    def forward(self, x, edge_index):
        
        if self.layers==1:
            mu = self.gnn_mu[0](x, edge_index)
            logvar = self.gnn_logvar[0](x, edge_index)
            return mu, logvar
        
        
        
        for i in range(self.layers):
            if i==0:
                mu = F.relu(self.gnn_mu[i](x, edge_index))
                logvar = F.relu(self.gnn_logvar[i](x, edge_index))
                continue
            elif i!=(self.layers-1) and i!=0:
                mu = F.relu(self.gnn_mu[i](mu, edge_index))
                logvar = F.relu(self.gnn_logvar[i](logvar, edge_index))
                continue
            else:
                mu = self.gnn_mu[i](mu, edge_index)
                logvar = self.gnn_logvar[i](logvar, edge_index)
                
        return mu, logvar


class MLP_Edge_Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, extra_layer=True):
        super(MLP_Edge_Decoder, self).__init__()
        self.extra_layer = extra_layer
        if self.extra_layer:
            self.layer_1 = nn.Linear(in_channels, hidden_channels)
            self.layer_2 = nn.Linear(hidden_channels, out_channels)
        else:
            self.layer_1 = nn.Linear(in_channels, out_channels)
            

    def forward(self, x):
        if self.extra_layer:
            x = F.relu(self.layer_1(x))
            x = self.layer_2(x)
        else:
            x = self.layer_1(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class Scenir(ARGVA):
    def __init__(self, encoder, discriminator, feature_decoder, edge_decoder=None):
        super(Scenir, self).__init__(encoder=encoder, discriminator=discriminator)
        
        # Feature Decoder: Input(latent_feature_tensor, edge_index) Output(initial_feature_tensor)
        self.feature_decoder = feature_decoder
        self.edge_decoder = edge_decoder
        
    def recon_loss_batched(self, x, pos_edge_index, all_edge_index, batch_vec):
        z = self.encode(x, pos_edge_index)
        
        feature_decoder_prediction = self.feature_decoder(z)
        feat_dec_loss = nn.MSELoss()(feature_decoder_prediction, x)
        
        if self.edge_decoder is None:
            z_dec = z
        else:
            z_dec = self.edge_decoder(z)
        
        pos_loss = -torch.log(
            self.decoder(z_dec, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)   # By including Self-Edges here, they won't be in negative samples

        neg_edge_index = batched_negative_sampling(all_edge_index_tmp, batch_vec, pos_edge_index.size(1))    # negative edges samples equal to number of positive edges
        neg_loss = -torch.log(1 - self.decoder(z_dec, neg_edge_index, sigmoid=True) + 1e-15).mean()

        return pos_loss + neg_loss + feat_dec_loss
    
    def latent_graph_embedding(self, x, edge_index):
        self.eval()
        z = self.encode(x.float(), edge_index).detach()
        return global_add_aggregation(z, edge_index, batch=None)


def get_model(config):
    # Define the model components
    encoder = multiGINEncoder(config['model_params']['embeddings_dim'], 
                              config['model_params']['latent_dim'], 
                              config['model_params']['latent_dim'], 
                              config['model_params']['encoder_layers'])
    discriminator = Discriminator(config['model_params']['latent_dim'], 
                                  config['model_params']['discriminator_hidden_dim'], 
                                  1)
    feature_decoder = MLP_Edge_Decoder(config['model_params']['latent_dim'], 
                                       config['model_params']['embeddings_dim'], 
                                       config['model_params']['embeddings_dim'])
    edge_decoder = MLP_Edge_Decoder(config['model_params']['latent_dim'], 
                                    config['model_params']['edge_decoder_hidden_dim'], 
                                    config['model_params']['edge_decoder_out_dim'])

    # Create the model
    model = Scenir(encoder, discriminator, feature_decoder, edge_decoder)
    
    return model