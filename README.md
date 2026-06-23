# Verso Nuovi Standard nell'Allineamento Multiplo di Sequenze (MSA)

Questo repository contiene il framework innovativo per l'Allineamento Multiplo di Sequenze (MSA) basato sulla combinazione di Deep Reinforcement Learning (DRL) e Algoritmi Genetici Multi-Obiettivo. Il sistema supera i limiti della precedente baseline GA-DPAMSA introducendo un'architettura neurale a convoluzioni dilatate (DCNN), strategie di Action Branching, moduli One-Shot e un orchestratore basato su NSGA-II per l'ottimizzazione simultanea sulla frontiera di Pareto delle metriche biologiche.  
🛠️ Data Engineering Pipeline

A differenza dei modelli precedenti addestrati su dati artificiali/sintetici , questa pipeline sposta il focus su dati biologici reali, permettendo all'agente di apprendere pattern evolutivi effettivi.  

La pipeline è suddivisa nei seguenti moduli (scripts/):

`ortho_data.py`: Interroga in modo automatizzato le API REST di OrthoDB v12, scaricando sequenze CDS (Coding DNA Sequences) appartenenti al clade Mammalia.  
`ortho_preparation.py` Attua una pipeline di preprocessing:
- Risoluzione Ambiguità: Identifica e mappa i caratteri nucleotidici non standard (es. 'N').  
- Rimozione Duplicati: Elimina le sequenze identiche all'interno dello stesso gruppo ortologo per evitare bias nel calcolo del Column Score.  
- Segmentazione (RandomCutSequence): Tronca le sequenze di lunghezza variabile in frammenti gestibili dalla finestra operativa fissa della rete.  

`cut_boards.py` Augmentation e data engineering: 
- Inietta stocasticamente gap e "rumore" controllati (RandomGapInsertion, RandomGapSubstitution) per creare lo "Stato 0" disallineato da sottoporre all'ambiente di training.  
- Mappa i nucleotidi in tensori numerici (A=1, T=2, C=3, G=4, Gap=5) e salva il dataset nel formato binario ad alte prestazioni HDF5 per azzerare i colli di bottiglia di I/O su GPU.  

🏗️ Architetture Proposte e Modelli

Il framework introduce tre principali pilastri architetturali per superare la complessità quadratica dei Transformer e la crescita esponenziale dello spazio delle azioni.  
1. DCNNMSA (Dueling Deep Q-Networks)

Modello di punta per la ricerca locale basato sull'interazione stepwise con l'environment di allineamento.  

- Encoder Convoluzionale Dilatato (DCNN): Sostituisce i moduli basati su self-attention (complessità O(L2)) utilizzando convoluzioni dilatate per catturare feature locali a lungo raggio in tempo lineare.  
- Double Dueling Q-Network: Separa la stima del valore dello stato (V) dai vantaggi di ciascuna azione (A) per stabilizzare l'apprendimento delle metriche.  
- Action Branching (BDQ): Risolve il problema dello spazio delle azioni esponenziale (2k−1) spacchettando l'output in sotto-rami di azione indipendenti, riducendo drasticamente la complessità del decisore.  

2. Allineamento One-Shot (PPO e GRPO)
Esplorazione di architetture avanzate pensate per abbattere drasticamente i tempi di inferenza rispetto agli approcci colonna-per-colonna (stepwise).  
- PPO (Proximal Policy Optimization): Ottimizzazione della policy con vincolo di clipping per generare l'allineamento finale in una singola inferenza complessiva.  
- GRPO (Group Relative Policy Optimization): Variante che calcola il vantaggio comparando un gruppo di output generati dallo stesso stato, riducendo l'overhead computazionale del network critico.  

3. Orchestratore Evolutivo NSGA-II
Algoritmo genetico multi-obiettivo che coordina la ricerca globale su allineamenti completi di dimensioni arbitrarie.  
- Dominanza Paretiana: Tratta il Sum-of-Pairs (SP) e il Column Score (CS) come obiettivi ortogonali e concorrenti, evitando lo sbilanciamento o il collasso di una metrica sull'altra tipico delle unioni forzate precedenti.  
- Mutazione Guidata come Local Search: Individua la sottomatrice (sub-board fissa, es. 3×30) con lo score peggiore all'interno dell'allineamento globale e la delega all'agente di RL (DCNNMSA) per un riallineamento ottimizzato localmente.  
- Selezione Elitista ed Estrazione: Sfrutta il Fast Non-Dominated Sorting e la Crowding Distance per mantenere diversificata la popolazione. L'output finale viene estratto dalla frontiera tramite scalarizzazione a posteriori pesata: SP + (CS × 100).  

📈 Risultati Sperimentali e Benchmark

I test prestazionali sono stati condotti in modalità zero-shot su un dataset di validazione di sequenze ortologhe di Mammalia non viste in fase di addestramento.  
1. Confronto con lo Stato dell'Arte (Tool Professionali)

I modelli proposti (DCNNMSA e l'ibrido NSGA-II-DCNN) sono stati messi a confronto con i principali benchmark di riferimento mondiali:  

| Modello | Tool	Column Score (CS) Medio	| Sum of Pairs (SP) Medio |
| --- | --- | --- |
| MSAPROBS | 0.491 | 73.08 |
|ClustalW	| 0.461	| 59.56 |
|ClustalOmega | 0.451 | 42.68 |
|DCNNMSA (Proposed)	| 0.441 | 46.24 |
|MAFFT	| 0.437 | 12.40 |
|MUSCLE	| 0.421	| 50.28 |
|NSGA-II-DCNN (Proposed) | 0.411 | 39.48 |
|PASTA | 0.366 | -8.58 |

Key Takeaways dai Risultati:
- Column Score: DCNNMSA si posiziona stabilmente nella fascia alta dei tool, superando software storici e consolidati come MAFFT, MUSCLE e PASTA.  
- Sum of Pairs: DCNNMSA ottiene un punteggio nettamente superiore rispetto a PASTA, MAFFT e ClustalOmega, dimostrando l'efficacia dei meccanismi di convoluzione e di reward biologico nel preservare la qualità globale.  

2. Impatto dell'Orchestratore (NSGA-II vs GA-Baseline)
- L'introduzione di NSGA-II ha eliminato la polarizzazione estrema delle soluzioni (alta varianza in cui l'algoritmo ottimizzava solo SP distruggendo il CS o viceversa).  
- Ha garantito una distribuzione dei punteggi molto più densa e coerente, elevando stabilmente la qualità media biologica della popolazione rispetto a GA-DPAMSA.

<img width="767" height="551" alt="image" src="https://github.com/user-attachments/assets/14ba477d-f446-4e8b-a759-7fa8fadd295f" />
<img width="754" height="530" alt="image" src="https://github.com/user-attachments/assets/70a05b16-526f-4869-9131-d2ef4bcd73b0" />

