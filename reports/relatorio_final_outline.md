# Relatório Final — Classificador de Cartas de Baralho por Visão Computacional

> **Disciplina:** Introdução à Ciência de Dados (ICD) — Curso de Análise e Desenvolvimento de Sistemas (ADS)
> **Item do projeto:** 2.6 (Relatório final, 4–8 páginas)
> **Autores:** Nicolas; Herick
> **Data:** _(preencher)_
> **Repositório:** _(preencher link do GitHub — ver Seção 7)_
>
> **Como usar este template:** cada seção traz, em blocos `> Orientação`, o que você deve preencher e quais documentos do repositório consultar. Substitua todos os campos marcados como **(preencher após o treino)** ou **(preencher)**. Não invente métricas: rode o treino no Colab e cole os números reais. Ao final, apague estes blocos de orientação se desejar uma versão "limpa".

---

## 1. Introdução e definição do problema

> **Orientação — preencher 2–4 parágrafos curtos:**
> - Apresente o problema: **classificação** de uma única carta de baralho já recortada em **53 classes** (52 cartas + 1 coringa/joker). Deixe claro que **detecção** (várias cartas em cena) é trabalho futuro, não faz parte do escopo.
> - Justifique a relevância com o **ângulo de impacto humano primário**: ferramenta **educacional** (ensino de probabilidade, regras de jogos e matemática para crianças e idosos; apoio à aprendizagem; também serve para ensinar visão computacional). Cite os públicos secundários: acessibilidade para pessoas com deficiência visual (ler a carta em voz alta) e pesquisa/benchmark em VC.
> - Enuncie o objetivo geral e os objetivos específicos (treinar um modelo de transfer learning, comparar com um baseline clássico, avaliar generalização OOD).
> - Referencie o documento de **definição do problema** (`docs/definicao.md`) para detalhes de motivação e requisitos.

**Contexto e motivação.** _(preencher)_

**Definição formal do problema.** Trata-se de um problema de **classificação supervisionada multiclasse** com 53 classes mutuamente exclusivas, onde a entrada é uma imagem RGB de 224×224 px de uma única carta já recortada e a saída é um único rótulo. _(preencher detalhes)_

**Objetivo geral.** _(preencher)_

**Objetivos específicos.**
- _(preencher — ex.: treinar EfficientNet-B0 via transfer learning)_
- _(preencher — ex.: comparar contra baseline HOG + Regressão Logística)_
- _(preencher — ex.: medir o gap de domínio em dados OOD)_

---

## 2. Dados e metodologia

> **Orientação — preencher e referenciar `docs/dados.md`:**
> - Descreva o **dataset principal**: "Cards Image Dataset-Classification" (autor *gpiosenka*, Kaggle), 53 classes, imagens 224×224×3 já recortadas, split pronto **train/valid/test = 7.624 / 265 / 265** (5 imagens por classe em validação e teste). Mencione a licença marcada como **"Other"** no Kaggle e a necessidade de **confirmar antes de redistribuir** (por isso `data/` contém apenas instruções, não as imagens).
> - Descreva o **conjunto OOD**: fotos de **1 baralho real próprio** que você vai fotografar, usado apenas para avaliação de generalização. Documente quantas fotos, condições de iluminação, fundo e dispositivo de captura.
> - Detalhe o **pré-processamento e data augmentation**: normalização ImageNet, resize/center-crop, e (no experimento de augmentation) flips/rotações/jitter de cor — descreva exatamente o que `src/data.py` aplica.
> - Detalhe os **modelos**: EfficientNet-B0 (Fase 1: backbone congelado / *feature extraction*; Fase 2: *fine-tuning* do topo com LR baixo) e o baseline HOG + Regressão Logística (scikit-learn). Referencie o **model card** (`docs/model_card.md`).

### 2.1 Dataset principal (Kaggle)

| Item | Valor |
|---|---|
| Nome | Cards Image Dataset-Classification (gpiosenka) |
| Nº de classes | 53 (52 cartas + 1 coringa) |
| Dimensão das imagens | 224 × 224 × 3 (RGB, já recortadas) |
| Split train / valid / test | 7.624 / 265 / 265 |
| Imagens por classe (val/test) | 5 / 5 |
| Licença | "Other" (Kaggle) — _(confirmar antes de redistribuir)_ |

_(preencher comentário sobre balanceamento e qualidade do dataset)_

### 2.2 Conjunto OOD (baralho real próprio)

> **Orientação:** descreva a coleta. Sugestão de campos a preencher: nº total de fotos, nº de cartas fotografadas, dispositivo (celular/câmera), condições (luz natural/artificial, fundo, ângulo). Reforce que serve **só para avaliação**, nunca para treino.

- Nº de fotos: **(preencher)**
- Dispositivo de captura: **(preencher)**
- Condições (luz/fundo/ângulo): **(preencher)**

### 2.3 Pré-processamento e data augmentation

- Normalização: médias/desvios do ImageNet. _(preencher confirmação)_
- Transformações base: _(preencher — resize/center-crop)_
- Augmentation (apenas no experimento 2): _(preencher — flip horizontal, rotação, color jitter, etc.)_

### 2.4 Modelos e treino

- **Modelo principal:** EfficientNet-B0 (torchvision), transfer learning em duas fases.
  - Fase 1 — *feature extraction*: backbone congelado, treina só a cabeça classificadora. _(preencher épocas/LR)_
  - Fase 2 — *fine-tuning*: descongela o topo, LR baixo. _(preencher épocas/LR)_
- **Baseline:** HOG + Regressão Logística (scikit-learn). _(preencher parâmetros do HOG e do classificador)_
- **Infra e reprodutibilidade:** Google Colab (GPU T4 gratuita), `set_seed(42)`, `requirements.txt` com versões pinadas. _(preencher batch size, otimizador, nº de épocas finais)_

---

## 3. Experimentos e resultados

> **Orientação — 3 experimentos planejados. NÃO invente números; rode no Colab e preencha:**
> - **Experimento 1 — Feature extraction (FE) vs Fine-tuning (FT):** mede o ganho de descongelar o topo.
> - **Experimento 2 — Com vs Sem data augmentation:** mede o impacto da augmentation na generalização.
> - **Experimento 3 — Avaliação OOD:** compara o desempenho no teste Kaggle contra as fotos do baralho real, medindo o **gap de domínio**.
> - Para cada experimento, preencha a tabela abaixo e comente os resultados. A accuracy **esperada** do transfer learning em cartas é ~93–95% (referência da literatura, **não** é resultado seu — substitua pelos números reais).
> - Inclua **matriz de confusão** e a análise de **"confusões perigosas"** (trocar naipe ou valor).

### 3.1 Tabela consolidada de resultados

> Métricas: **Accuracy** e **F1 macro** no teste Kaggle (in-distribution) + **Accuracy OOD** nas fotos do baralho real. Todos os valores abaixo são **placeholders (preencher após o treino)**.

| Configuração | Acc. (teste Kaggle) | F1 macro (teste Kaggle) | Acc. OOD (baralho real) | Observações |
|---|---|---|---|---|
| Baseline: HOG + Reg. Logística | _(preencher)_ | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| EfficientNet-B0 — Feature Extraction (FE) | _(preencher)_ | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| EfficientNet-B0 — Fine-Tuning (FT) | _(preencher)_ | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| EfficientNet-B0 — FT **sem** augmentation | _(preencher)_ | _(preencher)_ | _(preencher)_ | _(preencher)_ |

> _Referência da literatura (não é resultado deste trabalho): transfer learning em cartas tende a ~93–95% de accuracy in-distribution._

### 3.2 Experimento 1 — Feature extraction vs Fine-tuning

- Resultado: _(preencher)_
- Interpretação: _(preencher — o fine-tuning valeu o custo? quanto ganhou?)_

### 3.3 Experimento 2 — Com vs sem data augmentation

- Resultado: _(preencher)_
- Interpretação: _(preencher — a augmentation ajudou na generalização e/ou no OOD?)_

### 3.4 Experimento 3 — Avaliação OOD (gap de domínio)

- Gap medido (Acc. Kaggle − Acc. OOD): _(preencher)_
- Interpretação: _(preencher — quanto o desempenho cai fora da distribuição de treino e por quê)_

### 3.5 Matriz de confusão e confusões perigosas

> **Orientação:** insira a figura da matriz de confusão (exportada por `src/evaluate.py`) e discuta os erros mais frequentes. Classifique as **confusões perigosas**: trocar **naipe** (♠♥♦♣) ou **valor** (ex.: 6 vs 9, valete vs dama), que em um uso educacional ou de acessibilidade transmitiriam informação errada.

- Figura: _(inserir `reports/confusao_*.png`)_
- Confusões mais frequentes: _(preencher)_
- Confusões perigosas (naipe/valor): _(preencher)_

---

## 4. Discussão: limitações, ética e impacto humano

> **Orientação — referencie `docs/etica.md` e o model card. Cubra explicitamente os riscos levantados na pesquisa:**
> - **(a) Dual-use / trapaça:** a mesma tecnologia pode virar *Real-Time Assistance* (RTA) — proibido por plataformas como PokerStars e GGPoker — em jogos de azar e apostas. Declare que este é um **uso proibido** do projeto.
> - **(b) Jogo problemático:** risco de facilitar comportamento de jogo com dano social documentado.
> - **(c) Viés/limitação do dataset:** treinado em **um** tipo de baralho → tende a falhar em outros designs, iluminações e oclusão (conecte com o resultado do Experimento 3/OOD).
> - **(d) Privacidade (LGPD):** a câmera pode captar o ambiente e terceiros → minimização de dados, processamento **on-device**, sem armazenar imagens desnecessárias.
> - **(e) Documentação ética insuficiente:** publicar sem declarar usos proibidos é um risco em si → este relatório, o `docs/etica.md` e o model card mitigam isso.

### 4.1 Limitações técnicas

- _(preencher — escopo só de classificação, dependência de recorte prévio, dataset de 1 baralho, val/test pequenos com 5 img/classe)_
- _(preencher — ligar à queda observada no OOD)_

### 4.2 Considerações éticas

- **Dual-use e trapaça:** _(preencher — declarar uso proibido; citar RTA / PokerStars / GGPoker)_
- **Jogo problemático:** _(preencher)_
- **Viés do dataset:** _(preencher — ligar ao Experimento 3)_
- **Privacidade / LGPD:** _(preencher — minimização, on-device)_

### 4.3 Impacto humano

- **Primário — educacional:** _(preencher — ensino de probabilidade, regras, matemática; público infantil/idoso; ensino de VC)_
- **Secundário — acessibilidade:** _(preencher — leitura da carta em voz alta para deficientes visuais)_
- **Secundário — pesquisa/benchmark:** _(preencher)_

---

## 5. Conclusão e trabalhos futuros

> **Orientação — 1–2 parágrafos + bullets:**
> - Retome os objetivos e diga, com os números reais, se foram atingidos.
> - Resuma a principal lição de cada experimento (FE vs FT, augmentation, gap OOD).
> - Liste extensões futuras. **Cite explicitamente detecção com YOLO** (várias cartas em cena, do recorte único para a cena completa) como a evolução natural do escopo.

**Conclusão.** _(preencher — síntese com base nos resultados reais)_

**Trabalhos futuros.**
- **Detecção com YOLO:** estender de classificação de carta única para **detecção** de múltiplas cartas em cena (ex.: YOLOv8), removendo a dependência do recorte manual. _(preencher)_
- Ampliar o conjunto OOD com mais baralhos/designs e condições de captura. _(preencher)_
- Empacotar o modo educacional/acessibilidade (leitura em voz alta, exercícios de probabilidade). _(preencher)_
- _(preencher — outras extensões)_

---

## 6. Reprodutibilidade

> **Orientação:** descreva como reproduzir do zero. Referencie o `README.md` e a estrutura do `src/`. Reforce `set_seed(42)` e `requirements.txt` pinado.

1. Clonar o repositório: _(preencher comando, ver Seção 7)_
2. Criar ambiente e instalar dependências pinadas: `pip install -r requirements.txt`.
3. Obter os dados conforme `data/` (instruções) e `docs/dados.md` — o dataset **não** é redistribuído (licença "Other").
4. Configurar `src/config.py` (caminhos, hiperparâmetros) e garantir `set_seed(42)` (`src/seed.py`).
5. Baseline: `python -m src.baseline`. _(preencher)_
6. Treino: `python -m src.train` (ou o notebook `notebooks/treino_cartas_colab.ipynb` na GPU T4 do Colab).
7. Avaliação: `python -m src.evaluate` (gera métricas e matriz de confusão em `reports/`).
8. Predição em imagem única: `python -m src.predict --img <caminho>`. _(preencher)_

---

## 7. Repositório e instruções de acesso

- **Link do GitHub:** _(preencher — ex.: https://github.com/usuario/classificador-cartas)_
- **Licença do código:** MIT (ver `LICENSE`).
- **Documentos de apoio no repositório:**
  - `docs/definicao.md` — definição do problema (Seção 1)
  - `docs/dados.md` — dataset, OOD e licença (Seção 2)
  - `docs/etica.md` — riscos éticos e usos proibidos (Seção 4)
  - `docs/model_card.md` — model card do EfficientNet-B0 (Seções 2 e 4)
  - `README.md` — visão geral e reprodução (Seção 6)

---

## Apêndice A — Roteiro da apresentação (item 2.7, 10–15 min)

> **Orientação:** roteiro sugerido com tempos. Ajuste à duração-alvo e ensaie. Use as figuras de `reports/`.

| Bloco | Tempo | Conteúdo |
|---|---|---|
| 1. Abertura e problema | 1–2 min | O que é: classificar 1 carta recortada em 53 classes. Por que importa: ferramenta **educacional** (probabilidade/regras/matemática); acessibilidade e benchmark como secundários. |
| 2. Dados e método | 2–3 min | Dataset Kaggle (53 classes, 224×224, split 7.624/265/265); conjunto **OOD** (baralho real); EfficientNet-B0 (FE → FT) vs baseline HOG + Reg. Logística. |
| 3. Experimentos | 3–4 min | Os 3 experimentos: FE vs FT; com vs sem augmentation; **gap OOD**. Mostrar a tabela consolidada (números reais) e a **matriz de confusão**. |
| 4. Confusões perigosas e limitações | 1–2 min | Erros de naipe/valor; limitação de 1 baralho; queda no OOD. |
| 5. Ética e impacto | 2 min | Dual-use/RTA (uso **proibido**), jogo problemático, viés do dataset, privacidade/LGPD (on-device). Impacto educacional positivo. |
| 6. Conclusão e futuro | 1–2 min | O que foi atingido; **detecção com YOLO** como próximo passo. |
| 7. Perguntas | restante | Q&A. |

**Dicas de apresentação:** _(preencher — 1 slide por bloco, no máx. ~2 ideias por slide, mostrar 1 exemplo de predição correta e 1 confusão perigosa, ensaiar para fechar dentro de 15 min)._