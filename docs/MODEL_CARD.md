# Model Card — Classificador de Cartas de Baralho (EfficientNet-B0)

> Documento no estilo Hugging Face / Google Model Cards. Projeto acadêmico (ICD/ADS). As métricas reais devem ser preenchidas **após o treino no Google Colab** — todos os números abaixo marcados como "(preencher após o treino)" são *placeholders*.

---

## 1. Detalhes do Modelo

| Campo | Descrição |
|---|---|
| **Nome do modelo** | Classificador de Cartas de Baralho (Card Image Classifier) |
| **Versão** | 0.1.0 (pré-treino / esqueleto reprodutível) |
| **Tipo de tarefa** | Classificação de imagem — *single-label*, multiclasse |
| **Arquitetura** | Transfer learning com **EfficientNet-B0** (backbone pré-treinado em ImageNet) + cabeça de classificação linear adaptada para **53 classes** |
| **Framework** | PyTorch + torchvision |
| **Entrada** | 1 imagem RGB de **uma carta já recortada**, redimensionada para 224×224×3, normalizada com a média/desvio do ImageNet |
| **Saída** | Distribuição de probabilidade sobre 53 classes (52 cartas + 1 coringa/*joker*); rótulo previsto = `argmax` |
| **Classes (53)** | 13 valores (Ás, 2–10, Valete, Dama, Rei) × 4 naipes (copas, ouros, paus, espadas) + coringa/*joker* |
| **Estratégia de treino** | **Fase 1:** *feature extraction* (backbone congelado, treino apenas da cabeça). **Fase 2:** *fine-tuning* do topo da rede com *learning rate* baixo |
| **Modelo de baseline** | HOG + Regressão Logística (scikit-learn) — referência clássica para comparação |
| **Escopo** | **Apenas classificação** (1 carta recortada → 1 rótulo). **Detecção** (várias cartas em cena, localização) é trabalho futuro e **não** é suportada |
| **Autor / responsável** | Nícolas Pereira (nicolas.pereira@way2.com.br) — projeto acadêmico ICD/ADS |
| **Licença do código** | **MIT** (ver arquivo `LICENSE` do repositório) |
| **Licença dos dados** | Marcada como **"Other"** no Kaggle — **a confirmar antes de qualquer redistribuição** (ver Seção 4) |
| **Idioma da documentação** | Português do Brasil |
| **Data do documento** | 2026-06-01 |

---

## 2. Usos Pretendidos

O modelo foi concebido com **ângulo de impacto humano primariamente educacional**. Os usos a seguir são os endossados pelos autores.

### 2.1 Uso primário — Ferramenta educacional
- **Ensino de probabilidade e estatística:** reconhecer cartas para demonstrar cálculo de chances, combinações e raciocínio probabilístico de forma concreta e visual.
- **Ensino de regras de jogos e de matemática** para crianças e pessoas idosas, como apoio lúdico à aprendizagem.
- **Material didático para ensinar visão computacional:** o próprio pipeline (transfer learning, *data augmentation*, avaliação) serve de exemplo pedagógico de um projeto de classificação de imagens de ponta a ponta.

### 2.2 Usos secundários
- **Acessibilidade para pessoas com deficiência visual:** identificar uma carta e **lê-la em voz alta**, como apoio à autonomia em contextos não monetários (ex.: jogos recreativos, atividades educativas).
- **Pesquisa e *benchmark* em visão computacional:** estudar transfer learning, robustez a *out-of-distribution* (OOD) e *gaps* de domínio em um conjunto de classes bem definido.

### 2.3 Usuários previstos
Educadores, estudantes, pesquisadores de VC, desenvolvedores de ferramentas de acessibilidade e entusiastas de aprendizado de máquina.

---

## 3. Usos Fora de Escopo e Usos Proibidos

> Esta seção é **normativa**. O uso do modelo nas situações abaixo **viola a intenção dos autores** e, em vários casos, regras de plataformas e/ou legislação.

### 3.1 Usos PROIBIDOS (dual-use / dano social)
- **Apostas e jogos de azar com dinheiro real**, presencial ou *online*.
- **Real-Time Assistance (RTA)** — qualquer assistência em tempo real durante partidas com valor monetário. Ferramentas equivalentes são **expressamente proibidas** por operadoras como **PokerStars** e **GGPoker**, e seu uso configura **trapaça**.
- **Integração discreta / oculta** (formatos discretos: óculos, câmeras escondidas, *overlays* furtivos) destinada a obter vantagem indevida em jogos.
- **Contagem de cartas assistida por máquina** ou qualquer vantagem mecânica em cassinos e casas de aposta.
- Qualquer uso que **facilite ou agrave o jogo problemático** (dano social documentado associado ao comportamento de apostas).

### 3.2 Usos fora de escopo (não suportados tecnicamente)
- **Detecção/contagem de múltiplas cartas em uma cena** — o modelo só classifica **uma carta já recortada**.
- Leitura de cartas em **baralhos com design muito diferente** do conjunto de treino (ver Limitações, Seção 6).
- Decisões **automatizadas de alto risco** sem supervisão humana.
- Reconhecimento de objetos que **não sejam cartas de baralho**.

---

## 4. Dados de Treino

| Campo | Descrição |
|---|---|
| **Dataset** | *Cards Image Dataset-Classification* |
| **Autor** | **gpiosenka** (Kaggle) |
| **Link** | https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification |
| **Classes** | 53 (52 cartas + coringa/*joker*) |
| **Formato** | Imagens RGB **224×224×3, já recortadas** (uma carta por imagem) |
| **Split (pré-definido)** | **train: 7.624** / **valid: 265** / **test: 265** imagens |
| **Tamanho de val/test** | **5 imagens por classe** em validação e em teste (conjuntos pequenos — ver Limitações) |
| **Licença** | Marcada como **"Other"** no Kaggle. **Necessário confirmar os termos antes de redistribuir** os dados ou um modelo derivado. Este repositório **não redistribui** o dataset; a pasta `data/` contém **apenas instruções de download** |

### 4.1 Pré-processamento
- Redimensionamento para 224×224 (quando aplicável) e normalização com média/desvio do ImageNet.
- **Data augmentation** (experimento 2): transformações como *flip*, rotação leve, variação de brilho/contraste — aplicadas **somente no treino**.

---

## 5. Métricas de Avaliação

**Métricas adotadas:** *Accuracy* + **F1 macro** + **matriz de confusão** + **análise de "confusões perigosas"** (trocas de naipe ou de valor entre classes próximas).

> **Atenção:** todos os valores abaixo são *placeholders*. **(preencher após o treino)**. A *accuracy* esperada para transfer learning em cartas é de aproximadamente **93–95%**, mas esse número é uma expectativa de referência, **não** um resultado medido.

### 5.1 Resultados por experimento

| Experimento | Configuração | Accuracy | F1 macro | Observações |
|---|---|---|---|---|
| **Baseline** | HOG + Regressão Logística | (preencher após o treino) | (preencher após o treino) | Referência clássica |
| **Exp. 1a** | EfficientNet-B0, *feature extraction* (backbone congelado) | (preencher após o treino) | (preencher após o treino) | — |
| **Exp. 1b** | EfficientNet-B0, *fine-tuning* do topo | (preencher após o treino) | (preencher após o treino) | Comparar com 1a |
| **Exp. 2a** | Melhor modelo, **sem** *data augmentation* | (preencher após o treino) | (preencher após o treino) | — |
| **Exp. 2b** | Melhor modelo, **com** *data augmentation* | (preencher após o treino) | (preencher após o treino) | Comparar com 2a |

### 5.2 Avaliação OOD (Experimento 3)

| Conjunto de avaliação | Origem | Accuracy | F1 macro | *Gap* de domínio |
|---|---|---|---|---|
| **Teste Kaggle (in-distribution)** | Mesmo domínio do treino | (preencher após o treino) | (preencher após o treino) | — |
| **Fotos de baralho real (OOD)** | 1 baralho fotografado pelo usuário | (preencher após o treino) | (preencher após o treino) | (preencher após o treino) |

### 5.3 Análise qualitativa
- **Matriz de confusão:** (preencher após o treino) — inserir figura em `reports/`.
- **Confusões perigosas:** (preencher após o treino) — listar os pares de classes mais confundidos e destacar trocas de **naipe** (ex.: copas ↔ ouros) e de **valor** (ex.: 6 ↔ 9), que são as mais críticas em uso educacional/acessibilidade.

---

## 6. Limitações e Vieses

- **Treinado em um único tipo de baralho:** o dataset reflete **um design** de cartas. O desempenho tende a **degradar** em baralhos com estilos artísticos diferentes, cartas regionais, decks personalizados ou marcas distintas.
- ***Gap* de domínio (OOD):** as imagens de treino são **recortadas e padronizadas (224×224)**. Em fotos do mundo real (iluminação variável, fundo, sombras, perspectiva, oclusão parcial, desfoque, reflexos), espera-se **queda de acurácia** — quantificada no Experimento 3.
- **Conjuntos de validação e teste pequenos:** apenas **5 imagens por classe** (265 no total cada). Isso torna as estimativas de métrica **ruidosas** e sensíveis a poucos exemplos; intervalos de confiança são largos.
- **Escopo restrito a classificação:** não detecta nem segmenta múltiplas cartas; **exige recorte prévio** de uma única carta.
- **Viés de pré-treino (ImageNet):** o backbone carrega vieses do ImageNet, que não foi pensado para cartas.
- **Sem garantia de calibração:** as probabilidades de saída podem ser **mal calibradas**; não devem ser interpretadas como confiança real sem avaliação adicional.
- **Casos não cobertos:** cartas dobradas, sujas, parcialmente cobertas, em ângulos extremos ou sob iluminação atípica podem produzir **erros silenciosos**.

---

## 7. Considerações Éticas (resumo)

- **Dual-use / trapaça:** a mesma tecnologia pode ser usada para **RTA proibido** em pôquer/cassinos (PokerStars, GGPoker). Os usos proibidos estão declarados na Seção 3.
- **Jogo problemático:** há **dano social documentado** ligado a apostas; o projeto **não** endossa nem facilita uso em contextos de jogo por dinheiro.
- **Viés e limitação do dataset:** treinado em **um tipo de baralho** → risco de falha em outros designs, iluminações e oclusões (Seção 6). Resultados não devem ser generalizados além do domínio avaliado.
- **Privacidade (LGPD):** em uso com câmera, o dispositivo pode captar **o ambiente e terceiros**. Recomenda-se **minimização de dados**, **processamento *on-device*** sempre que possível, ausência de armazenamento desnecessário de imagens e **consentimento** quando houver pessoas identificáveis.
- **Transparência / documentação ética:** publicar o modelo **sem declarar os usos proibidos** é em si um risco. Este *model card* cumpre o papel de documentar explicitamente intenção, limites e restrições.
- **Supervisão humana:** ferramenta de **apoio**; decisões devem permanecer sob responsabilidade de uma pessoa, especialmente em contextos educacionais e de acessibilidade.

---

## 8. Como Reproduzir

### 8.1 Ambiente
- **Treino:** Google Colab com **GPU T4 gratuita**.
- **Dependências:** versões **fixadas** em `requirements.txt` (PyTorch, torchvision, scikit-learn, etc.).
- **Reprodutibilidade:** semente global fixa via `set_seed(42)` (`src/seed.py`) — controla `random`, `numpy` e `torch` (CPU/GPU).

### 8.2 Passos
1. **Obter os dados:** baixar o dataset *Cards Image Dataset-Classification* (gpiosenka) seguindo as instruções em `data/` (o repositório **não** redistribui as imagens).
2. **Instalar dependências:** `pip install -r requirements.txt`.
3. **Configurar:** ajustar caminhos e hiperparâmetros em `src/config.py`.
4. **Baseline:** treinar HOG + Regressão Logística com `src/baseline.py`.
5. **Treino principal:** executar `src/train.py` (ou o notebook `notebooks/treino_cartas_colab.ipynb`) — Fase 1 (*feature extraction*) e Fase 2 (*fine-tuning*).
6. **Avaliar:** `src/evaluate.py` gera accuracy, F1 macro, matriz de confusão e relatório de confusões perigosas em `reports/`.
7. **Inferência:** `src/predict.py` classifica uma carta recortada.
8. **OOD:** avaliar as **fotos do baralho real** do usuário e comparar com o teste Kaggle (Experimento 3).

### 8.3 Estrutura do repositório
```
src/        config.py, seed.py, data.py, model.py, baseline.py, train.py, evaluate.py, predict.py
notebooks/  treino_cartas_colab.ipynb
models/     pesos treinados (gerados)
data/       instruções de download (sem dados redistribuídos)
docs/       documentação
reports/    métricas, figuras, matriz de confusão
README.md   visão geral
requirements.txt  dependências fixadas
LICENSE     MIT (código)
```

---

## 9. Citação

```
@misc{classificador_cartas_2026,
  title  = {Classificador de Cartas de Baralho com EfficientNet-B0 (Transfer Learning)},
  author = {Pereira, Nicolas},
  year   = {2026},
  note   = {Projeto acadêmico ICD/ADS. Código sob licença MIT.
            Dataset: Cards Image Dataset-Classification (gpiosenka, Kaggle).}
}
```

---

*Model card sujeito a atualização após o treino. Métricas reais substituirão os placeholders "(preencher após o treino)". Em caso de dúvida sobre usos permitidos, prevalece a Seção 3 (Usos Fora de Escopo e Proibidos).*