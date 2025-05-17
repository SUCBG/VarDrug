import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VMAE(nn.Module):
    def __init__(self, original_emb_dim=768, chrom_dim=16, latent_dim=128, num_sift_classes=4, num_polyphen_classes=4):
        super(VMAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_sift_classes = num_sift_classes
        self.num_polyphen_classes = num_polyphen_classes
        
        input_dim = original_emb_dim + chrom_dim 

        self.initial_fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024, dropout=0.2),
            num_layers=2
        )
        
        self.sift_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_sift_classes)
        )
        
        self.polyphen_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_polyphen_classes)
        )
        
        self.downstream_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.2),
            num_layers=2
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder_expansion = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024, dropout=0.2),
            num_layers=2
        )
        
        self.decoder_sift_output = nn.Linear(512, num_sift_classes)
        self.decoder_polyphen_output = nn.Linear(512, num_polyphen_classes)
        
        self.chrom_embed = nn.Embedding(num_embeddings=24, embedding_dim=chrom_dim)

    def encode(self, original_emb, chrom):
        chrom_emb = self.chrom_embed(chrom)
        
        x = torch.cat([original_emb, chrom_emb], dim=-1)
        x = self.initial_fusion(x)
        x = x.unsqueeze(1)
        
        features = self.transformer(x)
        sift_logits = self.sift_head(features.squeeze(1))
        polyphen_logits = self.polyphen_head(features.squeeze(1))
        
        x = features.squeeze(1)
        x = self.downstream_fusion(x)
        x = x.unsqueeze(1)
        x = self.encoder_transformer(x).squeeze(1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, sift_logits, polyphen_logits

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_expansion(z)
        batch_size = x.size(0)
        tgt = torch.zeros(1, batch_size, 512).to(x.device)
        x = x.unsqueeze(0)
        x = self.decoder_transformer(tgt, x)
        x = x.squeeze(0)
        sift_logits = self.decoder_sift_output(x)
        polyphen_logits = self.decoder_polyphen_output(x)
        return sift_logits, polyphen_logits

    def forward(self, original_emb, chrom):
        z, mu, logvar, encoder_sift_logits, encoder_polyphen_logits = self.encode(original_emb, chrom)
        decoder_sift_logits, decoder_polyphen_logits = self.decode(z)
        return (decoder_sift_logits, decoder_polyphen_logits, 
                z, mu, logvar, 
                encoder_sift_logits, encoder_polyphen_logits)