# Dados

> As imagens **não** são versionadas no Git (ver `.gitignore`). Esta pasta guarda
> apenas as instruções de download, seguindo o princípio de ciência aberta
> ("dados sensíveis ou pesados não devem ir direto ao repositório").

## Dataset principal — Classificação (Kaggle)

**Cards Image Dataset-Classification** (autor: gpiosenka)
<https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification>

- **53 classes** = 52 cartas (13 valores × 4 naipes) + 1 coringa (*joker*).
- Imagens **224×224×3** (JPG), já recortadas (carta ocupa > 50% da imagem).
- Split pronto: **7.624 treino / 265 validação / 265 teste** (5 imagens por classe em val/teste).
- Estrutura: `train/`, `valid/`, `test/`, cada uma com 53 subpastas (uma por classe), + `cards.csv`.
- Licença: marcada como *"Other"* no Kaggle — **confirme a licença na página** antes de redistribuir.

### Como baixar

**Opção A — kagglehub (recomendada no Colab):**
```python
import kagglehub
path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
print(path)  # use este caminho como data_dir
```

**Opção B — Kaggle API (kaggle.json):**
1. Em <https://www.kaggle.com/settings> → *Create New API Token* (baixa `kaggle.json`).
2. No Colab, faça upload do `kaggle.json` e rode:
```bash
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d gpiosenka/cards-image-datasetclassification -p data/raw --unzip
```

Após baixar, aponte `data_dir` para a pasta que contém `train/`, `valid/` e `test/`
(por padrão o projeto usa `data/raw/cards`).

## Conjunto de teste OOD — seu baralho real (coleta manual)

Para o experimento de generalização (out-of-distribution), você vai fotografar
**um baralho real**. Siga o guia: [`../docs/guia_coleta_baralho_real.md`](../docs/guia_coleta_baralho_real.md).

Coloque as fotos em `data/raw/ood_baralho_real/`, com **uma subpasta por classe**
usando **exatamente os mesmos nomes** das subpastas do dataset do Kaggle
(ex.: `ace of spades/`, `king of hearts/`, `joker/`).
