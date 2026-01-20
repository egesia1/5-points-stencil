# Confronto Implementazioni Weak Scaling

## Riepilogo

Confronto tra due approcci per implementare weak scaling:
1. **Mia proposta**: `scripts/parametric_weak_scaling_CORRECTED.sh`
2. **Implementazione esistente**: `excluded/tavaa/run_weak_scaling.sbatch`

---

## Differenze Principali

### 1. Forma del Grid

| Nodes | Mia Proposta | Implementazione Esistente |
|-------|--------------|---------------------------|
| 1 | 16384×16384 (quadrato) | 10000×10000 (quadrato) |
| 2 | 23170×23170 (quadrato) | **20000×10000 (rettangolare 2:1)** |
| 4 | 32768×32768 (quadrato) | 20000×20000 (quadrato) |
| 8 | 46340×46340 (quadrato) | **40000×20000 (rettangolare 2:1)** |
| 16 | 65536×65536 (quadrato) | 40000×40000 (quadrato) |

**Implicazioni**:
- **Grid quadrato**: Migliore per decomposizione 2D bilanciata, comunicazione simmetrica
- **Grid rettangolare**: Può creare asimmetria nella comunicazione (più halo exchange in una direzione)

### 2. Scaling dell'Area

**Entrambe le implementazioni scalano correttamente l'area**:

| Nodes | Area Ratio | Work per Core |
|-------|------------|---------------|
| 1 | 1.00× | Baseline |
| 2 | 2.00× | Costante ✅ |
| 4 | 4.00× | Costante ✅ |
| 8 | 8.00× | Costante ✅ |
| 16 | 16.00× | Costante ✅ |

**Conclusione**: Entrambe mantengono work per core costante, requisito fondamentale per weak scaling.

### 3. Base Grid Size

- **Mia proposta**: 16384×16384 (2^14, potenza di 2)
  - Vantaggi: Allineamento memoria, cache-friendly, decomposizione più efficiente
- **Esistente**: 10000×10000 (valore arrotondato)
  - Vantaggi: Valore più semplice, già testato

### 4. Metodo di Calcolo

**Mia proposta**:
```bash
GRID_SIZE=$(python3 -c "import math; print(int(${BASE_GRID_SIZE} * math.sqrt(${nodes})))")
```
- ✅ Dinamico, facile estendere
- ⚠️ Richiede Python disponibile

**Esistente**:
```bash
case ${nodes} in
    1)  GRID_SIZE_X=10000; GRID_SIZE_Y=10000 ;;
    2)  GRID_SIZE_X=20000; GRID_SIZE_Y=10000 ;;
    # ...
esac
```
- ✅ Nessuna dipendenza esterna
- ✅ Più veloce (no calcoli runtime)
- ⚠️ Hardcoded, difficile estendere

### 5. Architettura dello Script

**Mia proposta (COMPLETA - parametric_weak_scaling_CORRECTED.sh)**:
- Script parametrico che genera job SLURM separati
- Testa **3 configurazioni × 3 energy sources × 5 node counts = 45 job**
- Ogni job è indipendente
- **⚠️ ECCESSIVO per test rapido in emergenza**

**Mia proposta (SEMPLIFICATA - quick_weak_scaling_PARAMETRIC.sh)**:
- Script che genera 5 job SLURM separati
- Testa **1 configurazione × 1 energy source × 5 node counts = 5 job**
- Ogni job è indipendente
- **✅ Adatto per test rapido**

**Esistente**:
- Singolo job SLURM che esegue tutti i test in sequenza
- Testa 1 configurazione (2×56) × 1 energy source × 5 node counts = **5 run in un job**
- Tutti i test nello stesso job
- **✅ Più semplice da monitorare (un solo job)**

---

## Vantaggi e Svantaggi

### Mia Proposta (`parametric_weak_scaling_CORRECTED.sh`)

**Vantaggi**:
- ✅ Grid sempre quadrato (migliore per decomposizione 2D)
- ✅ Potenza di 2 (16384 = 2^14, migliore per cache/alignment)
- ✅ Calcolo dinamico (facile aggiungere più nodi)
- ✅ Testa tutte le configurazioni e energy sources
- ✅ Job separati (se uno fallisce, gli altri continuano)

**Svantaggi**:
- ⚠️ Richiede Python per calcolo sqrt
- ⚠️ Genera molti job (45 totali)
- ⚠️ Più complesso da gestire

### Implementazione Esistente (`run_weak_scaling.sbatch`)

**Vantaggi**:
- ✅ **Già testata e funzionante**
- ✅ Nessuna dipendenza esterna (no Python)
- ✅ Valori hardcoded (più veloce)
- ✅ Singolo job (più semplice da monitorare)
- ✅ **Perfetta per test rapido in emergenza**

**Svantaggi**:
- ⚠️ Grid rettangolare per 2 e 8 nodi (può influenzare comunicazione)
- ⚠️ Valori non potenze di 2 (10000, 20000, 40000)
- ⚠️ Testa solo configurazione 2×56
- ⚠️ Difficile estendere (serve modificare case statement)

---

## Raccomandazione per Test Rapido

**Per una situazione di emergenza**, consiglio di usare l'**implementazione esistente** con queste modifiche:

1. **Adattare alla configurazione desiderata** (es. 8×14 invece di 2×56)
2. **Mantenere i valori hardcoded** per velocità
3. **Opzionale**: Cambiare base grid size a potenza di 2 se necessario

### Script Rapido Adattato

```bash
#!/bin/bash
#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH --job-name=weak_scaling_quick
#SBATCH --exclusive
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8    # Cambiare qui
#SBATCH --cpus-per-task=14      # Cambiare qui
#SBATCH -t 01:00:00
#SBATCH --mem=0

EXEC="../build/stencil_parallel"
N_STEPS=500
NODES_LIST="1 2 4 8 16"

RESULTS_DIR="../results/weak_scalability"
OUTPUT_FILE="${RESULTS_DIR}/weak_scalability_results_8x14.txt"
mkdir -p ${RESULTS_DIR}

module purge
module load openmpi/4.1.6--gcc--12.2.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

for nodes in ${NODES_LIST}; do
    TOTAL_TASKS=$((nodes * SLURM_NTASKS_PER_NODE))
    
    case ${nodes} in
        1)  GRID_SIZE_X=16384; GRID_SIZE_Y=16384 ;;
        2)  GRID_SIZE_X=23170; GRID_SIZE_Y=23170 ;;
        4)  GRID_SIZE_X=32768; GRID_SIZE_Y=32768 ;;
        8)  GRID_SIZE_X=46340; GRID_SIZE_Y=46340 ;;
        16) GRID_SIZE_X=65536; GRID_SIZE_Y=65536 ;;
    esac
    
    PROGRAM_ARGS="-x ${GRID_SIZE_X} -y ${GRID_SIZE_Y} -n ${N_STEPS} -e 1"
    
    echo "--- RUNNING: ${nodes} nodes, ${TOTAL_TASKS} tasks, Grid: ${GRID_SIZE_X}×${GRID_SIZE_Y} ---"
    srun -N ${nodes} -n ${TOTAL_TASKS} ${EXEC} ${PROGRAM_ARGS} >> ${OUTPUT_FILE}
done
```

**Modifiche rispetto all'originale**:
- Grid quadrato (X = Y) per tutti i nodi
- Valori calcolati come `16384 * sqrt(nodes)` (potenze di 2 quando possibile)
- Configurazione adattabile (8×14, 2×56, 16×7)

---

## Tabella Comparativa Completa

| Aspetto | Mia Proposta | Esistente | Per Emergenza |
|---------|--------------|-----------|---------------|
| **Grid Shape** | Sempre quadrato | Rettangolare (2,8 nodi) | ✅ Quadrato (adattato) |
| **Base Size** | 16384 (2^14) | 10000 | ✅ 16384 (adattato) |
| **Calcolo** | Python sqrt() | Hardcoded | ✅ Hardcoded |
| **Dipendenze** | Python | Nessuna | ✅ Nessuna |
| **Configurazioni** | 3 (8×14, 2×56, 16×7) | 1 (2×56) | ⚠️ 1 (adattabile) |
| **Energy Sources** | 3 (1, 2, 8) | 1 | ⚠️ 1 (sufficiente) |
| **Job Structure** | 45 job (completo) / 5 job (semplificato) | 1 job, 5 run | ✅ 1 job, 5 run |
| **Velocità Setup** | Lenta (45 job) | ✅ Rapida | ✅ Rapida |
| **Facilità Estensione** | ✅ Facile | ⚠️ Difficile | ⚠️ Media |

---

## Conclusione

**Per un test rapido in emergenza**:
1. Usa l'**architettura dell'implementazione esistente** (singolo job, valori hardcoded)
2. Adatta i valori del grid a **quadrato** e **potenze di 2** quando possibile
3. Testa una sola configurazione (es. 8×14 che è la migliore)
4. Usa 1 energy source (sufficiente per weak scaling)

Questo ti darà risultati validi in modo rapido, senza la complessità di 45 job separati.
