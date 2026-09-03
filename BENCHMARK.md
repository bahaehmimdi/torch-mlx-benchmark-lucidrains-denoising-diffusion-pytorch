# lucidrains/denoising-diffusion-pytorch — Benchmark torch-mlx sur Mac

**Statut : OK — 5/5 pass E2E, 671K params**

**torch-mlx** est une reimplémentation de l'API PyTorch basée sur le moteur **MLX
d'Apple Silicon**. L'architecture de **lucidrains/denoising-diffusion-pytorch** a été testée de manière **non modifiée**.

**torch-mlx** est une reimplémentation de l'API PyTorch basée sur le moteur **MLX d'Apple Silicon**. En testant des modèles réels non modifiés, **torch-mlx (mode compilé) a dépassé PyTorch MPS** sur la majorité des workloads GEMM/conv (transformers, ResNet, DCGAN, VAE). Exemples de gains vs PyTorch CPU : ResNet18 ~10×, ResNet50 ~15×, DCGAN ~15×, minGPT ~3.3×, nanoGPT ~2×, LSTM ~1.7×, VAE ~2.9×.

Un point central : **`mlx.core.compile` en mode lazy / compilé** attend de connaître toute la séquence d'opérations avant de lancer les calculs. C'est particulièrement important pour les **opérations de batching** : au lieu d'exécuter chaque petite opération GPU séparément (avec son overhead de dispatch/lancement à chaque itération), le mode compilé lazy construit d'abord le graphe d'opérations de tout le batch, le fusionne en kernels optimisés, puis l'exécute d'un seul coup. Pour un batch de N échantillons, l'overhead est amorti une seule fois au lieu de N fois — d'où des gains typiques de plusieurs fois (jusqu'à ~15×) dès que le travail par étape est suffisant.

## Compatibilité trouvée
Round 360 : Parameter.weight.data.copy_() (mx.array brut)
