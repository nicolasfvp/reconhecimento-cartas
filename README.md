# 🃏 Reconhecimento de Cartas de Baralho com IA

Projeto acadêmico (ICD/ADS) de **visão computacional**: treinar um modelo para
**classificar cartas de baralho** a partir de uma imagem (53 classes = 52 cartas + coringa),
usando *transfer learning* em PyTorch.

> **Status:** estrutura, código e documentação prontos. O treino é executado no
> **Google Colab** (ver [`notebooks/treino_cartas_colab.ipynb`](notebooks/treino_cartas_colab.ipynb)).
> As métricas reais serão preenchidas após o primeiro treino.

---

## 🎯 Objetivo e impacto humano

Construir uma **ferramenta educacional** que reconhece cartas de baralho comuns, útil para:

- **Educação** (foco principal): ensino de probabilidade, regras de jogos e matemática
  para crianças e idosos; e como material didático para ensinar visão computacional.
- **Acessibilidade** (secundário): ler a carta em voz alta para pessoas com deficiência visual,
  usando um baralho **comum** (sem cartas especiais/caras).
- **Pesquisa** (secundário): cartas são um *benchmark* didático para classificação,
  *data augmentation* e *transfer learning*.

> ⚠️ **Usos proibidos / fora de escopo:** este projeto **não** se destina a jogos de azar
> com dinheiro real, apostas, cassinos online, nem a assistência em tempo real durante
> partidas valendo dinheiro. Ver [`docs/03_etica_impacto.md`](docs/03_etica_impacto.md).

---

## 🗂️ Estrutura do repositório

```
projeto-IA/
├── src/                       # Código-fonte (pacote Python)
│   ├── config.py              # Configuração central (hiperparâmetros, caminhos)
│   ├── seed.py                # Reprodutibilidade (set_seed)
│   ├── data.py                # Dataloaders e transformações (PyTorch)
│   ├── model.py               # Transfer learning (EfficientNet-B0 / MobileNet / ResNet)
│   ├── baseline.py            # Baseline clássico (HOG + Regressão Logística)
│   ├── train.py               # Treino (feature extraction + fine-tuning, early stopping)
│   ├── evaluate.py            # Métricas, matriz de confusão, avaliação OOD
│   └── predict.py             # Inferência em uma imagem
├── notebooks/
│   └── treino_cartas_colab.ipynb   # Notebook de treino (Google Colab) — pipeline completo
├── models/                    # Checkpoints treinados (não versionados)
├── data/                      # Apenas instruções de download (ver data/README.md)
├── docs/                      # Documentos do trabalho (definição, dados, ética, model card)
├── reports/                   # Resultados, figuras e relatório final
├── requirements.txt
├── LICENSE                    # MIT
└── README.md
```

---

## 🧠 Abordagem

| Etapa | Método | Papel |
|-------|--------|-------|
| **Baseline** | HOG + Regressão Logística (scikit-learn) | linha de base barata (CPU) |
| **Modelo principal** | Transfer learning **EfficientNet-B0** (PyTorch) | melhor acurácia/custo |
| **Treino em 2 fases** | (1) backbone congelado → (2) *fine-tuning* do topo | generalização |

### Experimentos (item 2.4)

1. **Feature extraction (congelado) vs fine-tuning** — impacto na acurácia/overfitting.
2. **Com vs sem *data augmentation*** — efeito na generalização.
3. **Avaliação OOD** — teste do Kaggle vs **fotos de um baralho real** (mede o *gap* de domínio).

---

## 📊 Dataset

**[Cards Image Dataset-Classification (gpiosenka)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)**
— 53 classes, imagens 224×224 já recortadas, split pronto (7.624 / 265 / 265).
Detalhes e download em [`data/README.md`](data/README.md).

Para o experimento de generalização, você fotografará um baralho real seu — guia em
[`docs/guia_coleta_baralho_real.md`](docs/guia_coleta_baralho_real.md).

---

## 🚀 Como reproduzir

### Opção 1 — Google Colab (recomendado, com GPU grátis)

1. Suba este repositório para o GitHub.
2. Abra [`notebooks/treino_cartas_colab.ipynb`](notebooks/treino_cartas_colab.ipynb) no Colab.
3. `Ambiente de execução → Alterar o tipo de hardware → GPU (T4)`.
4. Edite a variável `REPO_URL` na primeira célula com o link do seu repositório.
5. Execute as células em ordem (setup → dados → baseline → treino → avaliação → experimentos → OOD).

### Opção 2 — Local (CPU; instalar Python)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Baseline (rápido, CPU):
python -m src.baseline --data-dir data/raw/cards --max-per-class 80

# Treino (lento em CPU; recomendado usar o Colab):
python -m src.train --data-dir data/raw/cards --backbone efficientnet_b0

# Predição em uma imagem:
python -m src.predict caminho/para/carta.jpg --checkpoint models/efficientnet_b0_best.pt
```

> **Reprodutibilidade:** `set_seed(42)` em todo o pipeline, `requirements.txt` com versões
> fixadas e split fixo. Documente a GPU usada (T4/P100) no relatório.

---

## 📈 Resultados *(preencher após o treino)*

| Modelo | Acurácia (teste) | F1 macro | Acurácia OOD |
|--------|------------------|----------|--------------|
| Baseline (HOG + LogReg) | _(preencher)_ | _(preencher)_ | — |
| EfficientNet-B0 (feature extraction) | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| EfficientNet-B0 (fine-tuning) | _(preencher)_ | _(preencher)_ | _(preencher)_ |

Acurácia esperada do *transfer learning* em cartas: **~93–95%** (referência da literatura).

---

## ⚠️ Limitações

- Modelo treinado em **um único tipo de baralho/estilo** → pode falhar com outros designs,
  iluminação ruim, oclusão ou fundos complexos (ver experimento OOD).
- Conjuntos de validação/teste do dataset são pequenos (5 imagens/classe) → métricas ruidosas.
- Classificação assume **uma carta por imagem**; não detecta múltiplas cartas na cena
  (detecção via YOLO fica como **trabalho futuro**).

---

## 🔭 Trabalhos futuros

- Detecção de múltiplas cartas na cena (YOLOv8/YOLO11).
- Coleta de dataset mais diverso (vários baralhos, iluminações, oclusão).
- App de leitura por voz (modo assistivo) com processamento *on-device*.

---

## 📚 Documentação

- [Definição do problema e requisitos](docs/01_definicao_problema.md) (item 2.1)
- [Dados e preparação](docs/02_dados.md) (item 2.2)
- [Avaliação ética e impacto](docs/03_etica_impacto.md) (item 2.5)
- [Model Card](docs/MODEL_CARD.md)
- [Guia de coleta do baralho real (OOD)](docs/guia_coleta_baralho_real.md)
- [Esqueleto do relatório final](reports/relatorio_final_outline.md) (itens 2.6 e 2.7)

---

## 📄 Licença

Código sob licença [MIT](LICENSE). O dataset do Kaggle possui licença própria
(marcada como *"Other"*) — confirme antes de redistribuir as imagens.
