# Dataset de decisões de Aronian

## Fonte e coleta

Fonte pública: [PGN Mentor](https://www.pgnmentor.com/files.html), arquivo [Aronian.zip](https://www.pgnmentor.com/players/Aronian.zip). A coleta realizada em 9 de novembro de 2026 preservou a carreira disponível de 1993 a 2026. Os metadados completos estão em [dataset_report.json](../reports/aronian_v1/dataset_report.json).

| Divisão | Partidas | Decisões de Aronian |
| --- | ---: | ---: |
| Treino | 4.491 | 204.952 |
| Validação | 552 | 24.877 |
| Teste | 557 | 25.741 |
| Total | 5.600 | 255.570 |

São 129.326 decisões de brancas e 126.244 de pretas. Dos 5.609 registros lidos após normalizar a formatação, nove com o nome `Aronian,Laurel` foram excluídos. Uma partida foi recuperada corrigindo uma linha em branco entre cabeçalhos e movimentos, sem alterar jogadas. Não foram encontradas duplicatas pelo critério aplicado.

## Preparação

O script `scripts/build_dataset.py` reconstrói cada partida pela linha principal e extrai somente os turnos de Levon Aronian. Os nomes são normalizados e comparados com aliases explícitos; não basta conter o sobrenome.

Partidas com erros de leitura, posições iniciais inválidas, lances ilegais ou nulos são rejeitadas integralmente. O critério de duplicação considera jogadores normalizados, data, posição inicial e sequência completa, ignorando comentários e diferenças no nome do evento.

A divisão aproximada de 80/10/10 usa um hash determinístico da sequência completa, com semente `aronian-v1`. Uma partida não cruza divisões, e sequências completas idênticas permanecem juntas. A divisão não é cronológica e não garante posições inéditas: aberturas e outras posições podem aparecer em partidas diferentes.

## Formato e arquivos

Cada linha de `train.jsonl.gz`, `validation.jsonl.gz` ou `test.jsonl.gz` contém:

- `fen`: posição antes da jogada, na orientação original.
- `move_uci` e `move_san`: lance real que a rede deve prever.
- `player_id`, `color`, `game_id` e `split`: identificação e divisão.
- `ply`: meio-lance contado desde a posição inicial do registro; `fullmove_number`: contador do tabuleiro.

`games.jsonl.gz` preserva a posição inicial, os cabeçalhos e todos os movimentos UCI, inclusive os dos adversários. Eles ajudam a reconstruir as posições, mas não são respostas a aprender nos arquivos de decisões. Para obter o histórico anterior a um exemplo, aplique `moves_uci[:ply - 1]` a `initial_fen`. Não forneça movimentos futuros ou o resultado final como entrada.

`report.json` reúne as contagens; `excluded.jsonl` registra exclusões; `sample.jsonl` fornece uma amostra; `verification.json` confirma a reconstrução e conferência de todas as decisões. `aronian.data` cria o cache compacto a partir desses arquivos, com hashes de origem e versão da codificação.

## Reprodução e limites

Na raiz do projeto, depois de instalar as dependências completas do README:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py
.\.venv\Scripts\python.exe scripts\verify_dataset.py
.\.venv\Scripts\python.exe -m aronian.data
```

Os PGNs e o cache não estão no Git. O download é reutilizado quando `data/raw/Aronian.zip` existe. A reconstrução substitui os arquivos processados. Para processar um PGN local de Aronian em outra pasta:

```powershell
.\.venv\Scripts\python.exe scripts\build_dataset.py --pgn "C:\caminho\partidas.pgn" --output data/processed/aronian_extra
.\.venv\Scripts\python.exe scripts\verify_dataset.py --data data/processed/aronian_extra
```

A fonte pode atualizar seu ZIP. Para repetir a mesma coleção, é necessária uma cópia do arquivo original, cujo SHA-256 é `d9fa17c7a3dab9c76627e93d00bf0c6f70dbe6deb12289b71a4c34ec4601fcc5`. O SHA-256 do PGN original é `fac0122d3c55f8a2b09cb79dc00b1e9a906a059267fe6f5dafa0a7416898c430`. Um novo download pode produzir outras contagens; compare o relatório com os metadados entregues. A disponibilidade gratuita do arquivo não foi tratada como licença de redistribuição.

A base inclui partidas da infância e diferentes ritmos, sem filtro por obtenção do título de GM. Nenhuma partida aceita contém a tag `TimeControl`, portanto o ritmo é desconhecido nesse campo. A coleção não garante cobertura completa da carreira, e os cabeçalhos podem conter imprecisões. A FEN não inclui o histórico de repetição.
