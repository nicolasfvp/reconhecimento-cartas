# ✅ Guia de Passos Humanos

Checklist das etapas que precisam de ação humana para concluir o projeto.
(As etapas de código/estrutura já estão prontas no repositório.)

🔗 Repositório: <https://github.com/nicolasfvp/reconhecimento-cartas> (público)
👥 Grupo: Nicolas e Herick

---

- [x] **1. Criar o repositório no GitHub e dar push**
  Repositório criado e código + documentação enviados.

- [x] **2. Tornar o repositório público**
  Necessário para clonar no Colab sem token e compartilhar o link.

- [x] **3. Configurar acesso ao Kaggle (`kaggle.json`)**
  Token criado para baixar o dataset *Cards Image Dataset-Classification* (gpiosenka).

- [x] **4. Rodar o notebook no Google Colab (GPU T4)**
  `notebooks/treino_cartas_colab.ipynb` executado: setup → dataset → EDA → baseline → treino → avaliação → experimentos.

- [ ] **5. Fotografar 1 baralho real (conjunto de teste OOD)**
  Seguir `docs/guia_coleta_baralho_real.md` (nomes de pasta exatos + nota de privacidade/LGPD).
  Rodar de novo a célula de avaliação OOD do notebook para medir o *gap* de domínio.

- [ ] **6. Preencher as métricas reais**
  Substituir os campos *"(preencher após o treino)"* em `README.md`, `docs/MODEL_CARD.md` e
  `reports/relatorio_final_outline.md` com os números obtidos no Colab. Depois: novo commit/push.

- [ ] **7. Confirmar a licença exata do dataset gpiosenka**
  No Kaggle a licença está marcada como *"Other"* — confirmar antes de redistribuir imagens.

- [ ] **8. Escrever o relatório final (PDF) e a apresentação (10-15 min)**
  Usar o template `reports/relatorio_final_outline.md` (já traz a estrutura + roteiro da apresentação).
  Exportar o relatório em PDF para a entrega.

---

> Última atualização: 2026-06-01 — passos 1 a 4 concluídos.
