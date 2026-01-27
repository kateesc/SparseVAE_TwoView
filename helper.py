import sys
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from evaluation import evaluate
from sklearn.metrics import silhouette_score
from scipy.stats import chi2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 21

class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_metric = float('inf')
        self.max_accuracy = 0

    def early_stop(self, validation_metric):
        if validation_metric < self.min_validation_metric:
            self.min_validation_metric = validation_metric
            self.counter = 0
            self.max_accuracy = 0
        elif validation_metric > (self.min_validation_metric + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def perform_kmeans(final_embedding, labels, n_clusters):
    seeds = [0, 12, 21, 42, 1234]
    acc, nmi, silhouette = [], [], []

    em = final_embedding.detach().cpu().numpy()

    for seed in seeds:
        km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=seed)
        y_pred = km.fit_predict(em)

        nmi_, ari_, f_score_, acc_ = evaluate(labels, y_pred)
        silhouette_ = silhouette_score(em, y_pred, metric="euclidean")

        acc.append(acc_)
        nmi.append(nmi_)
        silhouette.append(silhouette_)

    return np.mean(acc), np.std(acc), np.mean(nmi), np.std(nmi), np.mean(silhouette), np.std(silhouette)


def perform_knn(final_embedding, labels, n_clusters):
    seeds = [0, 12, 21, 42, 1234]
    acc = []

    X = final_embedding.detach().cpu().numpy()

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.25, random_state=seed, stratify=labels
        )
        knn = KNeighborsClassifier(n_neighbors=n_clusters)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))

    return np.mean(acc), np.std(acc)


class InstanceLoss(nn.Module):
    """https://github.com/ChenyuxinXMU/MOCSS/blob/main/mocss/code/contrastive_loss.py"""
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # sim = torch.tensor(cosine_similarity(z.detach().numpy())) / self.temperature
        sim = torch.matmul(z, z.T) / self.temperature
        # np.savetxt('../../data/BRCA/sim_matric.txt', sim.detach().numpy())
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # selected_sim = self.select_sim_negative(sim, self.batch_size)
        # negative_samples = selected_sim[self.mask].reshape(N, -1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

def evaluate_final_embedding(final_embedding,y_test,n_clusters):
    kmeans_acc_mean,kmeans_acc_std,kmeans_nmi_mean,kmeans_nmi_std = perform_kmeans(final_embedding, y_test,n_clusters)
    knn_acc_mean,knn_acc_std = perform_knn(final_embedding,y_test,n_clusters)
    print("Kmeans nmi mean-std: ",kmeans_nmi_mean, kmeans_nmi_std,"  KNN accuracy mean-std: ", knn_acc_mean,knn_acc_std)
    return kmeans_nmi_mean, kmeans_nmi_std, knn_acc_mean, knn_acc_std

def get_sigma_params(sigmas_init, disease):
    """
    Generic sigma prior settings for any dataset key (including 'plant').
    The original repo hard-coded 'brca'/'kirc' only; this makes it usable for plants.
    """
    sigmas_init = np.asarray(sigmas_init, dtype=float)

    # Reasonable, stable defaults:
    sig_quant = 0.1
    sig_df = 10

    # Use a low quantile of observed stds to set scale, with floor to avoid 0
    base = max(np.quantile(sigmas_init, q=0.10), 1e-3)
    sig_scale = (base ** 2) * chi2.ppf(1 - sig_quant, sig_df) / sig_df
    return sig_df, sig_scale
