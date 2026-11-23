# Roteiro: treinar uma rede de Aronian e usá-la em uma aplicação web

**Documento histórico de planejamento.** A execução adotada foi local, e o treinamento já foi concluído. Para reproduzir a versão entregue, use [TREINAMENTO_LOCAL.md](../TREINAMENTO_LOCAL.md). As propostas abaixo não são uma descrição dos recursos efetivamente implementados.

Planejamento de 9 de novembro de 2026. Este documento define o notebook a implementar; ainda não há treinamento executado nem modelo exportado.

## 1. Escolha do ambiente

O Colab é tecnicamente adequado para um primeiro modelo pequeno em PyTorch. Porém, sua [FAQ oficial](https://research.google.com/colaboratory/faq.html) lista “chess training” entre as atividades restritas em runtimes gerenciados gratuitos sem saldo positivo de unidades de computação. A mesma FAQ informa que essas restrições podem ser removidas com plano pago e saldo positivo.

Por isso, planejar o treino de xadrez para **Colab pago com saldo positivo**, ou executar o mesmo código em Jupyter/local com recursos próprios. Não assumir que o plano gratuito seja apropriado. Não contratar recursos automaticamente como parte da implementação.

GPU, memória e duração da sessão variam. O notebook deve detectar o hardware, medir o tempo de uma época e salvar checkpoints persistentes. Não prometer duração ou custo antes dessa medição. O Colab será o ambiente de treinamento; a aplicação web terá execução independente.

## 2. Resultado pretendido

Primeira versão: uma rede de política pequena, inicializada aleatoriamente, que prevê os lances de Levon Aronian e pode escolher um lance legal durante uma partida. Treino supervisionado exclusivamente no dataset já preparado. Sem pesos pré-treinados, autojogo, busca de árvore ou rede de valor nesta primeira experiência.

Fluxo: dataset → preparação → treinamento → validação → teste final → pacote de inferência → aplicação web.

O resultado será um modelo experimental de imitação. Acerto na previsão, fidelidade de estilo e força de jogo são propriedades diferentes; nenhuma classificação Elo será prometida sem avaliação apropriada.

## 3. Dados de entrada existentes

Diretório local: `data/processed/levon_aronian/`.

| Divisão | Partidas | Decisões |
| --- | ---: | ---: |
| Treino | 4.491 | 204.952 |
| Validação | 552 | 24.877 |
| Teste | 557 | 25.741 |

Total: 5.600 partidas e 255.570 decisões. Arquivos processados ocupam aproximadamente 8 MB comprimidos. Levar também `games.jsonl.gz`, `report.json` e `verification.json`.

Preservar as divisões existentes. A mesma partida e a mesma sequência completa não cruzam divisões, mas posições de partidas diferentes podem se repetir. Não apresentar o teste como composto exclusivamente de posições inéditas.

A coleta cobre a carreira de 1993 a 2026, incluindo infância e diferentes ritmos. Nenhuma partida aceita fornece a tag `TimeControl`. A versão inicial representará essa mistura; recortes por período poderão ser experimentos posteriores, escolhidos pela validação.

## 4. Contrato de representação da versão 1

Definir e versionar este contrato antes de treinar. O site deverá usar exatamente o mesmo contrato.

### Entrada

Tensor `float32` de forma `[batch, 19, 8, 8]`:

- Canais 0–5: peões, cavalos, bispos, torres, damas e reis brancos.
- Canais 6–11: as mesmas peças pretas, na mesma ordem.
- Canal 12: constante 1 quando as brancas jogam; 0 quando as pretas jogam.
- Canais 13–16: direitos de roque branco curto, branco longo, preto curto, preto longo, como planos constantes de 0/1.
- Canal 17: casa de en passant da FEN, marcada com 1; zero nas demais casas.
- Canal 18: relógio de meios-lances, `min(halfmove_clock, 100) / 100`, como plano constante.

Orientação fixa: `row = rank - 1`, `column = file`, de modo que a1 seja `[0,0]` e h8 seja `[7,7]`. Não girar posições de pretas nem espelhar tabuleiros na versão 1. Não usar resultado final, movimentos futuros, identidade do adversário ou nome do evento como entrada.

Esta entrada não representa o histórico completo nem a contagem de repetições. O controlador de partidas deverá manter o histórico para aplicar as regras de empate. Acrescentar histórico à rede exigirá uma nova versão do contrato e novo treinamento.

### Saída e ações

Usar uma codificação simples e completa, independente dos lances que apareceram no dataset:

`action_id = ((from_square * 64 + to_square) * 5) + promotion_code`

Casas: a1=0, b1=1, ..., h8=63. Promoção: 0=nenhuma, 1=cavalo, 2=bispo, 3=torre, 4=dama. Total: **20.480 saídas**. Muitas combinações são impossíveis; a máscara de legalidade elimina essas opções.

Roques usam o movimento UCI do rei no xadrez padrão; en passant usa origem/destino do peão. A codificação inclui subpromoções e não depende de conhecer o conjunto de teste.

O modelo retorna pontuações brutas. Antes da entropia cruzada ou da escolha do lance, conservar apenas as ações correspondentes aos lances legais. Em posição terminal, não aplicar softmax sobre uma máscara vazia: retornar o resultado da partida.

### Arquitetura inicial proposta

- Convolução 3×3 de 19 para 64 canais, normalização e ReLU.
- Quatro blocos residuais, cada um com duas convoluções 3×3 de 64 canais, normalização e ReLU.
- Cabeça de política: convolução 1×1 de 64 para 320 canais.
- Saída `[batch, 320, 8, 8]`, reorganizada para `[batch, 8, 8, 320]` e depois `[batch, 20480]`. Cada casa de origem possui 64 destinos × 5 códigos de promoção.

Essa reorganização precisa corresponder exatamente à fórmula de `action_id`. A arquitetura é uma hipótese inicial de engenharia, não uma escolha já validada. Ela evita uma camada totalmente conectada muito grande e usa operações comuns para facilitar exportação.

## 5. Roteiro das células do notebook

| Bloco | Trabalho | Saída/critério |
| --- | --- | --- |
| 1. Apresentação | Explicar objetivo, ambiente apropriado e limites do experimento | Usuário identifica o que será treinado |
| 2. Ambiente | Instalar dependências compatíveis, verificar CUDA/GPU, registrar versões e seed | Hardware e versões registrados |
| 3. Persistência | Configurar pasta de execução no Drive ou diretório persistente local | Destino de checkpoints definido |
| 4. Importação | Receber um ZIP do dataset ou copiá-lo do Drive para o disco da sessão | Arquivos e hashes conferidos |
| 5. Inspeção | Exibir contagens, cores, datas e algumas posições com seus lances | Conferência visual e rótulos legais |
| 6. Codificação | Implementar os 19 canais, ações e conversão reversa para UCI | Casos de brancas/pretas, roque, en passant e promoções passam |
| 7. Cache | Preparar posições compactas, rótulos e listas de ações legais | Pré-processamento feito uma única vez |
| 8. Referência | Medir acerto esperado de seleção uniforme entre lances legais na validação | Referência simples de comparação |
| 9. Rede | Instanciar CNN aleatória e exibir parâmetros e dimensões | Saída correta e valores finitos |
| 10. Diagnóstico | Treinar brevemente em 128–256 exemplos de treino e verificar queda do erro | Erros de rótulo/gradiente investigados antes do treino maior |
| 11. Treino completo | Reinicializar a rede e o otimizador; treinar com validação por época | Curvas e checkpoints persistentes |
| 12. Seleção | Escolher o checkpoint pela menor perda de validação | Uma configuração final congelada |
| 13. Teste | Avaliar uma vez o modelo escolhido | Relatório final sem reajustes orientados pelo teste |
| 14. Inferência | Carregar os pesos em CPU e gerar lances legais | Execução independente da GPU de treino |
| 15. Exportação | Exportar ONNX e comparar com PyTorch | Paridade numérica e decisões verificadas |
| 16. Pacote | Gerar ZIP de inferência e checkpoint separado para retomada | Arquivos salvos no Drive e disponíveis para download |

O notebook usará funções dos módulos do projeto para evitar manter duas implementações diferentes de entrada, rede e saída. Será possível importar o notebook como `.ipynb` no Colab.

### Cuidados de memória e desempenho

Copiar os dados para o disco da sessão antes de treinar; usar o Drive para persistência. Não reler PGNs/FENs e recalcular legalidade em cada época. Salvar listas compactas de índices legais e construir a máscara apenas para cada lote: guardar 255 mil máscaras densas de 20.480 posições desperdiçaria memória.

Guardar os canais binários em `uint8` ou representação compacta e converter para `float32` no lote. O canal normalizado do relógio será reconstruído a partir de seu valor. Não colocar todo o dataset na GPU. A representação de cache também terá versão e hashes do dataset.

## 6. Configuração inicial do treino

Valores propostos para a primeira experiência; ajustes dependerão da validação:

| Parâmetro | Valor inicial |
| --- | --- |
| Otimizador | AdamW |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Batch size | 256; reduzir para 128 ou 64 se necessário |
| Máximo de épocas | 30 |
| Early stopping | 5 épocas sem melhora relevante da perda de validação |
| Função de perda | Entropia cruzada com máscara de lances legais |
| Mistura dos exemplos | Embaralhar somente treino |
| Precisão inicial | float32; testar precisão mista se o hardware justificar |
| Seleção do modelo | Menor perda de validação |

Uma época significa uma passagem pelas 204.952 decisões de treino. Medir a primeira época e estimar o restante com base no hardware realmente recebido. Não estabelecer uma meta de acurácia sem experimentar.

Salvar dois estados diferentes:

- `best_weights.pt`: pesos do melhor modelo pela validação, para inferência.
- `last_checkpoint.pt`: pesos atuais, otimizador, scheduler se usado, época, estado do early stopping, geradores aleatórios e scaler se usado, para continuar um treinamento interrompido.

Persistir a cada época e também em intervalos de tempo caso uma época seja longa. Usar gravação temporária seguida de substituição para reduzir risco de checkpoint incompleto. A retomada de um checkpoint intermediário pode recomeçar a época se a posição do amostrador não tiver sido persistida; deixar esse comportamento explícito.

Registrar configuração, versões, histórico de métricas, seed e hashes dos dados. Conservar o conjunto de teste fora das decisões sobre arquitetura e hiperparâmetros.

## 7. Avaliação

- Perda de validação/teste: qualidade das probabilidades atribuídas aos lances observados.
- Top-1: frequência com que a primeira escolha coincide com o lance real.
- Top-3: frequência com que o lance real está entre as três primeiras escolhas legais.
- As mesmas métricas por cor e após `ply > 20`, para reduzir a influência dos primeiros dez lances completos.
- Número de partidas e decisões em cada recorte; não apresentar recortes pequenos como evidência forte.
- Referência uniforme legal: média de `1 / número_de_lances_legais` para top-1, sem usar informação futura.
- Latência em CPU, tamanho dos pesos e tempo de carregamento, medidos no ambiente real.

Como complemento, jogar partidas completas contra um jogador humano e um oponente simples. Verificar que o sistema aplica todos os movimentos legalmente, detecta posições terminais e respeita a política documentada de reivindicação de empate. Vitórias não substituem as métricas de imitação.

Para uma avaliação posterior mais rigorosa, criar um protocolo que exclua posições vistas no treino. Não mudar silenciosamente o protocolo depois de observar os resultados do teste atual.

## 8. Exportação e download

Pacote de inferência proposto:

```text
aronian_model_v1.zip
  best_weights.pt
  model.onnx
  config.json
  encoding.json
  model.py
  encoding.py
  predict.py
  requirements-inference.txt
  metrics.json
  parity_cases.json
  README.md
```

Salvar pesos com `state_dict` e carregar a arquitetura correspondente. Um arquivo de pesos sozinho não descreve integralmente a arquitetura nem a interpretação das entradas e saídas. O [PyTorch documenta esse fluxo](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

Exportar em modo `eval`, inicialmente em float32, com nome da entrada, nome da saída e dimensões documentados. Escolher uma versão ONNX/opset compatível com as versões fixadas do exportador e do runtime de destino. Validar o modelo exportado com ONNX Runtime em CPU, usando posições de ambas as cores e casos especiais. Comparar pontuações com tolerâncias explícitas e investigar diferenças de classificação, inclusive empates numéricos.

O arquivo ONNX conterá a rede. Geração de lances legais, máscara, escolha do lance e controle das regras permanecerão no código da aplicação.

Criar o ZIP, salvar uma cópia persistente e disponibilizar download pelo notebook. O checkpoint de retomada fica separado, pois contém estados adicionais que a aplicação web não precisa carregar.

Critério de entrega: descompactar o pacote em um ambiente novo, carregar sem a sessão original do Colab e obter uma jogada legal a partir de uma posição de teste. Exportação só será marcada como concluída depois da verificação de paridade.

## 9. Aplicação web

### Primeira integração recomendada: servidor Python

1. Navegador mostra o tabuleiro e recebe a jogada humana.
2. Servidor mantém a partida ou reconstrói a posição a partir da posição inicial e do histórico UCI validado.
3. Biblioteca de regras verifica o movimento e as condições de término.
4. Encoder transforma a posição no tensor contratado.
5. Rede retorna pontuações; aplicação mascara lances ilegais.
6. Aplicação escolhe um lance, atualiza a partida e devolve o movimento UCI ao navegador.

Carregar o modelo uma vez ao iniciar o serviço, em modo de avaliação. Começar com CPU e medir a latência antes de contratar GPU para inferência. O servidor pode usar os pesos PyTorch ou ONNX Runtime.

Contrato sugerido para um serviço sem estado: receber `initial_fen` e `moves_uci`; validar e reconstruir o histórico; responder com `move_uci`, FEN resultante e estado da partida. Em uma API com sessões, manter esse histórico no servidor. Uma FEN isolada não basta para determinar repetições de posição.

Seleção inicial: escolher a maior probabilidade legal. Posteriormente oferecer amostragem com temperatura calibrada na validação. Aumentar a aleatoriedade não deve ser apresentado automaticamente como maior fidelidade ao estilo.

### Alternativa posterior: executar no navegador

O [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) permite inferência em JavaScript. WASM é uma opção inicial de compatibilidade; WebGPU pode ser avaliado conforme navegador e operadores suportados.

Esse caminho exige portar o encoder e o mapeamento de ações para JavaScript, comparar os tensores com Python e testar o runtime no navegador. Manter as regras da partida e o histórico no cliente. Usar um Web Worker para evitar bloquear a interface. A vantagem é dispensar servidor de inferência; a contrapartida é depender do dispositivo e disponibilizar os pesos aos usuários.

Não afirmar compatibilidade no navegador somente porque o arquivo passou no ONNX Runtime Python: executar os mesmos casos de paridade no runtime web.

## 10. Ordem de implementação e marcos

1. **Preparação:** empacotar dataset, implementar encoder/decoder e verificar legalidade e casos especiais.
2. **Notebook executável:** ambiente, cache, CNN, diagnóstico pequeno, treino, retomada e gráficos.
3. **Experimento:** executar no ambiente escolhido, selecionar pela validação e medir o teste final.
4. **Modelo portátil:** pacote PyTorch/ONNX validado fora do ambiente de treino.
5. **Partida web:** integrar primeiro ao servidor Python e testar partidas completas.
6. **Execução no navegador, se desejada:** portar codificação e verificar paridade JavaScript/ONNX.

Concluir a primeira versão quando o notebook conseguir retomar o treinamento, o pacote puder ser baixado e carregado de forma independente e a aplicação puder concluir partidas com movimentos legais. Melhorar força e fidelidade ao estilo será uma etapa experimental posterior, guiada por métricas.
