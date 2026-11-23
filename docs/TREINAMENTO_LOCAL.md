# Treinamento local e reprodução

O experimento entregue foi concluído: 14 épocas, 44.842 atualizações e melhor perda de validação na época 9. Os pesos selecionados e relatórios estão em `runs/aronian_v1/`. Uma reprodução deve usar outra pasta para preservar esses arquivos.

## 1. Ambiente

Na raiz do projeto, use Python 3.13 e instale as dependências completas:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

Também é possível executar `instalar.ps1` se a política do PowerShell permitir. Não é necessário ativar a `.venv`: os comandos usam diretamente seu executável.

O ambiente original usou Python 3.13.7, PyTorch 2.15.0+cpu, Intel Core i7-6700HQ, aproximadamente 16 GB de RAM e quatro threads. A GTX 950M não participou do treinamento. Os planos do cache ocupam aproximadamente 301 MiB; reserve pelo menos 1 GB livre para os dados processados e artefatos, além do ambiente Python. O tempo varia com o computador e a configuração.

## 2. Dataset e cache

Os dados brutos e o cache não acompanham o Git. Estes comandos baixam o ZIP público, extraem as decisões de Aronian, conferem todas as partidas e preparam os tensores:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py
.\.venv\Scripts\python.exe scripts\verify_dataset.py
.\.venv\Scripts\python.exe -m aronian.data
```

O ZIP em `data/raw/Aronian.zip` é reutilizado quando já existe. O cache em `data/cache/aronian_v1/` também é reutilizado se os hashes e a codificação coincidirem; se houver incompatibilidade, use outro `--output` na preparação do cache e passe o mesmo caminho em `--data` ao treinamento e à avaliação. Detalhes e limites de reprodução estão em [DATASET.md](DATASET.md).

## 3. Treinar uma rede nova

```powershell
.\.venv\Scripts\python.exe -m aronian.training --run runs/reproducao --epochs 30
```

A pasta `runs/reproducao` precisa estar vazia ou não existir. Para outro experimento, escolha um nome diferente. O comando inicializa a rede do zero; não carrega os pesos entregues.

A configuração está em `configs/local_cpu.json`: 32 canais, dois blocos residuais, lotes de 64, AdamW com learning rate de 0,001 e weight decay de 0,0001. Cada época percorre o treino embaralhado e depois mede a validação sem atualizar os parâmetros. O teste não participa da seleção do modelo.

O melhor modelo é escolhido pela menor perda de validação, com melhora mínima de 0,0001. O treinamento para após cinco épocas consecutivas sem essa melhora ou ao atingir o limite de épocas. O valor de `--epochs` é o total desejado, não uma quantidade adicional.

O atalho PowerShell executa a coleta, verificação quando necessária, cache e treinamento, usando `runs/reproducao` por padrão:

```powershell
.\treinar.ps1 -Execucao runs/reproducao -Epocas 30
```

Se scripts `.ps1` forem bloqueados, use os comandos Python anteriores.

## 4. Interromper e retomar

Pressione **Ctrl+C uma vez** e aguarde a mensagem de salvamento. Na etapa de validação, a parada pode aguardar sua conclusão. Depois:

```powershell
.\.venv\Scripts\python.exe -m aronian.training --run runs/reproducao --resume --epochs 30
```

Ou:

```powershell
.\treinar.ps1 -Execucao runs/reproducao -Retomar -Epocas 30
```

O checkpoint salva pesos, otimizador, ordem dos exemplos, próximo lote, métricas acumuladas, geradores aleatórios e estado da parada antecipada. Ele é gravado aproximadamente a cada 120 segundos e nos limites das épocas. O encerramento forçado pode perder o trabalho posterior ao último salvamento. Na retomada, a configuração e o cache devem corresponder à execução anterior; apenas o total de épocas pode mudar.

O checkpoint da execução original não está no Git. Por isso, depois de clonar o projeto, não use `--resume` com `runs/aronian_v1`; essa pasta fornece os pesos para inferência e os relatórios finais.

## 5. Acompanhar e avaliar

| Arquivo da execução | Finalidade |
| --- | --- |
| `best_weights.pt` | Pesos e arquitetura selecionados pela validação |
| `last_checkpoint.pt` | Estado para continuar o treinamento |
| `history.json` e `learning_curve.svg` | Métricas por época e curva de perda |
| `config.json` e `environment.json` | Configuração, versões e identificação do cache |
| `training_summary.json` | Resumo do treinamento concluído |
| `test_metrics.json` | Avaliação final, gerada pelo comando abaixo |

Depois de escolher a configuração pela validação, avalie uma vez no teste:

```powershell
.\.venv\Scripts\python.exe -m aronian.evaluate --run runs/reproducao
```

Top-1 mede a coincidência da primeira opção com o lance de Aronian; top-3 mede se ele está entre as três primeiras. Há recortes por cor e depois dos primeiros dez lances completos. Os resultados não são uma estimativa de Elo. Ajustar repetidamente o modelo com base no teste compromete sua função de avaliação final.

## 6. Jogar e exportar

O jogo padrão usa os pesos entregues. Para experimentar uma reprodução ou exportá-la:

```powershell
.\.venv\Scripts\python.exe -m aronian.web --weights runs/reproducao/best_weights.pt --open
.\.venv\Scripts\python.exe -m aronian.export --run runs/reproducao --output exports/reproducao
```

Encerre o servidor com Ctrl+C antes de executar o segundo comando na mesma janela. A pasta de exportação deve estar vazia. A exportação produz um ZIP com pesos, código de inferência, configuração, relatórios e ONNX. A equivalência PyTorch/ONNX é conferida em CPU; a execução em JavaScript exige validação própria. `--skip-onnx` gera somente o pacote PyTorch.

Também existe o modo de jogo no terminal:

```powershell
.\.venv\Scripts\python.exe -m aronian.predict --weights runs/reproducao/best_weights.pt --play
```

Informe movimentos UCI, como `e2e4` ou `a7a8q`. O controlador reivindica empates disponíveis automaticamente. Para determinar repetições corretamente em inferência, forneça a posição inicial e o histórico completo, não apenas uma FEN isolada.

## 7. Verificação rápida sem treino completo

Os testes automatizados não exigem o download dos PGNs:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Depois de preparar o dataset, é possível conferir o fluxo em uma amostra, usando nomes de pastas ainda não utilizados:

```powershell
.\.venv\Scripts\python.exe -m aronian.data --limit 128 --output data/cache/diagnostico
.\.venv\Scripts\python.exe -m aronian.training --data data/cache/diagnostico --run runs/diagnostico --epochs 1 --max-steps 1
.\.venv\Scripts\python.exe -m aronian.training --data data/cache/diagnostico --run runs/diagnostico --epochs 1 --resume
.\.venv\Scripts\python.exe -m aronian.evaluate --data data/cache/diagnostico --run runs/diagnostico
```

Esses pesos são marcados como `diagnostic_only` e não representam o resultado final do trabalho. Na implementação, o ensaio interrompido e retomado produziu pesos e métricas idênticos ao ensaio contínuo.
