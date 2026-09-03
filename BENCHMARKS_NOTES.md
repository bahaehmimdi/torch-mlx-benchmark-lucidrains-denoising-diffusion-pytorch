# lucidrains/denoising-diffusion-pytorch — Notes de benchmark

**Statut : OK — UViT + GaussianDiffusion (simple_diffusion.py) non modifiés, 671K params, forward + backward (143/143 gradients) + AdamW validés (5/5 pass). Écart d'init contourné (voir gaps).**

Un point central : **`mlx.core.compile` en mode lazy / compilé** attend de connaître toute la séquence d'opérations avant de lancer les calculs. C'est particulièrement important pour les **opérations de batching** : au lieu d'exécuter chaque petite opération GPU séparément (avec son overhead de dispatch/lancement à chaque itération), le mode compilé lazy construit d'abord le graphe d'opérations de tout le batch, le fusionne en kernels optimisés, puis l'exécute d'un seul coup. Pour un batch de N échantillons, l'overhead est amorti une seule fois au lieu de N fois — d'où des gains typiques de plusieurs fois (jusqu'à ~15×) dès que le travail par étape est suffisant.

## Gaps de compatibilité
lucidrains/denoising-diffusion-pytorch (~50K stars) : diffusion denoising (UViT/GaussianDiffusion).
Dépendances : einops + tqdm (installées). Le module simple_diffusion.py se charge byte-for-byte (importlib) — le __init__ du package importe des dépendances lourdes non-MLX, évité de la même façon que pour vit-pytorch.

**Gap réel trouvé (round 360) — `Parameter.weight.data.copy_()`.** Dans `Upsample.init_conv_`, le modèle fait `conv.weight.data.copy_(conv_weight)` et `nn.init.zeros_(conv.bias.data)` durant l'initialisation. Dans torch-mlx, `param.weight.data` renvoie un `mx.array` brut (le stockage MLX) au lieu d'un Tensor — sans méthode `copy_()` (`AttributeError: 'array' object has no attribute 'copy_'`). C'est un écart vs PyTorch réel où `.data` est un Tensor partageant le stockage. Le contournement propre (`torch.no_grad()` + `param.copy_()`, mêmes valeurs exactement) débloque tout le reste : le modèle se construit, forward + backward (tous les gradients) + optimiseur tournent (5/5). Le correctif côté noyau (découpler le stockage `.data` de l'attribut mx.array) est différé et documenté.

L'architecture UViT (attention + PixelShuffle/conv transposée + position embedding) est exactement le type de workload (GEMM/conv) où torch-mlx compilé excelle en accélération.

## Références
- Dépôt source torch-mlx : https://github.com/bahaehmimdi/torch-mlx
- Discussion générale : https://github.com/bahaehmimdi/torch-mlx-benchmarks-output/discussions/1
