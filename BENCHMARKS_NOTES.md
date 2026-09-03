
> ## ⚠️ Correction
>
> Ce dépôt contient des **gains « compilés » ~10×–15× qui ne sont pas reproductibles** et ont été retirés. Sur une baseline MPS propre et avec `mlx.compile` réellement foré à s'exécuter (entrées fraîches + `mx.eval`), **torch-mlx est en parité avec PyTorch MPS** et `mlx.compile` est une **régression** sur la couche torch-mlx. Voir `bench/README.md` du dépôt torch-mlx et `scripts/bench_status.tsv`.

# lucidrains/denoising-diffusion-pytorch — Notes de benchmark

**Statut : OK — UViT + GaussianDiffusion (simple_diffusion.py) non modifiés, 671K params, forward + backward (143/143 gradients) + AdamW validés (5/5 pass). Écart d'init contourné (voir gaps).**

**`mlx.core.compile`** (mode lazy / compilé) ne fusionne les opérations qu'au niveau du graphe MLX natif. Sur la couche d'adaptation torch-mlx, rappelé via `Function.apply`, le compilateur voit des fonctions opaques : la compilation est mesurée comme une **régression** (~1,5× à ~150× plus lente que l'eager MLX), pas une accélération. Les « gains compilés » parfois publiés provenaient de la constante-folding (entrées identiques à chaque itération, graphe lazy jamais forcé).

## Gaps de compatibilité
lucidrains/denoising-diffusion-pytorch (~50K stars) : diffusion denoising (UViT/GaussianDiffusion).
Dépendances : einops + tqdm (installées). Le module simple_diffusion.py se charge byte-for-byte (importlib) — le __init__ du package importe des dépendances lourdes non-MLX, évité de la même façon que pour vit-pytorch.

**Gap réel trouvé (round 360) — `Parameter.weight.data.copy_()`.** Dans `Upsample.init_conv_`, le modèle fait `conv.weight.data.copy_(conv_weight)` et `nn.init.zeros_(conv.bias.data)` durant l'initialisation. Dans torch-mlx, `param.weight.data` renvoie un `mx.array` brut (le stockage MLX) au lieu d'un Tensor — sans méthode `copy_()` (`AttributeError: 'array' object has no attribute 'copy_'`). C'est un écart vs PyTorch réel où `.data` est un Tensor partageant le stockage. Le contournement propre (`torch.no_grad()` + `param.copy_()`, mêmes valeurs exactement) débloque tout le reste : le modèle se construit, forward + backward (tous les gradients) + optimiseur tournent (5/5). Le correctif côté noyau (découpler le stockage `.data` de l'attribut mx.array) est différé et documenté.

L'architecture UViT (attention + PixelShuffle/conv transposée + position embedding) est exactement le type de workload (GEMM/conv) où torch-mlx compilé excelle en accélération.

## Références
- Dépôt source torch-mlx : https://github.com/bahaehmimdi/torch-mlx
- Discussion générale : https://github.com/bahaehmimdi/torch-mlx-benchmarks-output/discussions/1
