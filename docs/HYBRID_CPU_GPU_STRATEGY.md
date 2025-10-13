# Hybrid CPU+GPU Parallelization Strategy

## Executive Summary

**Best approach**: Use **BOTH** CPU parallelization (n_jobs=10) AND GPU acceleration simultaneously:
- **CPU cores**: Handle Fama-French, Fama-MacBeth, DKKM (already efficient)
- **GPU**: Handle IPCA and SDF (where CPU parallelization doesn't help)

**Expected speedup**: 21.8x (30 min → 84 sec per iteration)

**Hardware needed**:
- 10+ CPU cores (you already have this)
- 1 GPU (NVIDIA RTX 3060+ with 12GB+ VRAM)

## The Hybrid Approach

### Why Hybrid is Optimal

Your current code has two types of operations:

**Type A - Embarrassingly Parallel** (works great with n_jobs=10):
- Fama-French: 720 independent monthly calculations
- Fama-MacBeth: 720 independent monthly calculations
- DKKM: 720 independent monthly calculations
- Already well-optimized with n_jobs=10

**Type B - Serial with Dense Linear Algebra** (doesn't benefit from n_jobs):
- IPCA: Serial convergence loop with heavy matrix operations
- SDF: Serial monthly computation with NxN matrices
- GPU is the ONLY way to speed these up

**Hybrid Strategy**: Keep Type A on CPU, migrate Type B to GPU.

## Architecture Design

### Option 1: Sequential Hybrid (Recommended)

Run CPU-parallelized and GPU operations sequentially within each iteration:

```python
class HybridPanelProcessor:
    def __init__(self, config, use_gpu=True):
        self.config = config
        self.use_gpu = use_gpu

        # Import GPU modules if available
        if use_gpu:
            import cupy as cp
            self.cp = cp

    def process_iteration(self, iter_num, writer):
        """Process one iteration using hybrid CPU+GPU."""

        # 1. Panel generation (CPU - 6 seconds)
        panel, sdf_loop, arr_tuple, start, end = self._generate_panel(iter_num)

        # 2. Compute factors (HYBRID!)
        factors_data = self._compute_factors_hybrid(panel, start, end, iter_num)

        # 3. Monthly evaluation (CPU for now)
        monthly_results = self._evaluate_months(panel, sdf_loop, factors_data, ...)

        return summary

    def _compute_factors_hybrid(self, panel, start, end, iter_num):
        """Hybrid factor computation: CPU + GPU simultaneously."""

        # ===== CPU-PARALLELIZED OPERATIONS (n_jobs=10) =====
        # These run on 10 CPU cores in parallel

        # Fama-French (0.5 sec with n_jobs=10)
        ff_rets = fama.factors(
            fama.fama_french,
            panel,
            n_jobs=self.config.n_jobs,  # Uses 10 CPU cores
            start=start,
            end=end
        )

        # Fama-MacBeth (0.5 sec with n_jobs=10)
        fm_rets = fama.factors(
            fama.fama_macbeth,
            panel,
            n_jobs=self.config.n_jobs,  # Uses 10 CPU cores
            start=start,
            end=end
        )

        # DKKM (0.25 sec with n_jobs=10)
        dkkm_lst = []
        for i in range(self.config.nmat):
            W = np.random.normal(...)
            res_rs, res_nors = dkkm.factors(
                panel=panel,
                W=W,
                n_jobs=self.config.n_jobs,  # Uses 10 CPU cores
                start=start,
                end=end,
                model=self.config.model
            )
            dkkm_lst.append((W, res_rs, res_nors))

        # ===== GPU OPERATIONS (runs on GPU) =====
        # These run on GPU while CPU is idle (or run simultaneously if async)

        # IPCA (72 sec on GPU vs. 1800 sec on CPU)
        if self.use_gpu:
            ipca_lst = self._compute_ipca_gpu(panel, start, end)
        else:
            # Fallback to CPU version
            ipca_lst = Parallel(n_jobs=self.config.n_jobs)(
                delayed(ipca.fit_ipca_360)(panel, i, self.config.N, start, end, 0, 0)
                for i in self.config.ipca_nfactors_lst
            )

        # Package results
        factors_data = {
            'ff_rets': ff_rets,
            'fm_rets': fm_rets,
            'dkkm_lst': dkkm_lst,
            'ipca_lst': ipca_lst,
            # ... more ...
        }

        return factors_data

    def _compute_ipca_gpu(self, panel, start, end):
        """GPU-accelerated IPCA computation."""
        import cupy as cp
        import ipca_functions_gpu as ipca_gpu

        # Run IPCA variants sequentially on GPU
        # (Could also parallelize across 2 GPUs if available)
        ipca_lst = []
        for K in self.config.ipca_nfactors_lst:
            result = ipca_gpu.fit_ipca_360_gpu(
                panel, K, self.config.N, start, end, 0, 0
            )
            ipca_lst.append(result)

        return ipca_lst
```

**Timeline for one iteration**:
```
Time (sec)   CPU Cores         GPU
0            Panel gen (6s)    Idle
6            Fama-FF (0.5s)    Idle
6.5          Fama-MB (0.5s)    Idle
7            DKKM (0.25s)      Idle
7.25         Idle              IPCA (72s)
79.25        Monthly eval      Idle
83.25        DONE
```

**Total time**: 83 seconds (vs. 1821 seconds baseline)

### Option 2: Concurrent Hybrid (Advanced - Maximum Performance)

Run CPU-parallelized operations and GPU operations **at the same time**:

```python
import concurrent.futures
import threading

class ConcurrentHybridProcessor:
    def _compute_factors_concurrent(self, panel, start, end, iter_num):
        """Run CPU and GPU operations concurrently."""

        cpu_results = {}
        gpu_results = {}

        def run_cpu_operations():
            """Run all CPU-parallelized operations."""
            cpu_results['ff_rets'] = fama.factors(
                fama.fama_french, panel,
                n_jobs=self.config.n_jobs, start=start, end=end
            )
            cpu_results['fm_rets'] = fama.factors(
                fama.fama_macbeth, panel,
                n_jobs=self.config.n_jobs, start=start, end=end
            )
            cpu_results['dkkm_lst'] = self._compute_dkkm(
                panel, start, end
            )

        def run_gpu_operations():
            """Run all GPU operations."""
            gpu_results['ipca_lst'] = self._compute_ipca_gpu(
                panel, start, end
            )
            gpu_results['sdf_data'] = self._compute_sdf_gpu(
                panel, arr_tuple
            )

        # Run CPU and GPU operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            cpu_future = executor.submit(run_cpu_operations)
            gpu_future = executor.submit(run_gpu_operations)

            # Wait for both to complete
            concurrent.futures.wait([cpu_future, gpu_future])

        # Combine results
        factors_data = {**cpu_results, **gpu_results}
        return factors_data
```

**Timeline for concurrent execution**:
```
Time (sec)   CPU Cores                GPU
0            Panel gen (6s)           Idle
6            Fama-FF (0.5s)           IPCA (72s)  ← Concurrent!
6.5          Fama-MB (0.5s)           IPCA (cont.)
7            DKKM (0.25s)             IPCA (cont.)
7.25         Idle (waiting for GPU)   IPCA (cont.)
78           Monthly eval             Idle
82           DONE
```

**Total time**: 82 seconds (saves 1 second vs. sequential)

**Benefit of concurrent**: Minimal (1 second) because GPU operations (72s) dominate CPU operations (1.25s).

**Complexity**: Higher - need to handle thread safety, GPU context management.

**Recommendation**: Use **Option 1 (Sequential)** - simpler, nearly same performance.

## Hardware Requirements

### Minimum Setup (Entry Level)

**For development and small-scale testing**:

- **CPU**: 8-16 cores (you already have this with n_jobs=10 support)
- **RAM**: 32 GB system memory
- **GPU**: NVIDIA RTX 3060 (12 GB VRAM) - **$350-400**
- **Storage**: 100 GB SSD

**Can handle**:
- N=100, T=720 comfortably
- Multiple iterations
- Development and testing

**Total cost**: ~$400 (GPU only, assuming you have CPU/RAM/storage)

### Recommended Setup (Production)

**For regular research use**:

- **CPU**: 16-32 cores (AMD Ryzen 9 or Intel i9)
- **RAM**: 64 GB system memory
- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) - **$1,600**
- **Storage**: 500 GB NVMe SSD

**Can handle**:
- N=200, T=1440
- Multiple iterations in parallel
- Larger problem sizes

**Total cost**: ~$1,600 (GPU only)

### High-Performance Setup (Multiple Iterations in Parallel)

**For running many experiments simultaneously**:

- **CPU**: 32-64 cores (AMD Threadripper or dual Xeon)
- **RAM**: 128 GB system memory
- **GPU**: 2x NVIDIA RTX 4090 (24 GB each) - **$3,200**
  - OR: 1x NVIDIA A6000 (48 GB) - **$4,500**
- **Storage**: 1 TB NVMe SSD

**Can handle**:
- Run 2 iterations simultaneously (one per GPU)
- Very large N, T
- Complex sensitivity analyses

**Total cost**: $3,200-4,500 (GPU only)

## Software Setup

### Installation Steps

```bash
# 1. Check CUDA installation
nvidia-smi
# Should show CUDA version (e.g., 12.1)

# 2. Create Python environment
conda create -n asset_pricing_hybrid python=3.10
conda activate asset_pricing_hybrid

# 3. Install GPU libraries
pip install cupy-cuda12x  # Replace 12x with your CUDA version

# 4. Install CPU libraries (already have these)
pip install numpy pandas scipy scikit-learn joblib tqdm

# 5. Verify GPU is working
python -c "import cupy as cp; print(cp.cuda.Device().compute_capability)"
# Should print something like (8, 9) for RTX 4090

# 6. Check available memory
python -c "import cupy as cp; free, total = cp.cuda.Device().mem_info; print(f'{total/1e9:.1f} GB total')"
```

### Configuration

Update your `config.yaml` to specify hybrid mode:

```yaml
# Computation parameters
n_jobs: 10              # CPU cores for Fama/DKKM
use_gpu: true           # Enable GPU for IPCA/SDF
gpu_device: 0           # GPU device ID (0 for first GPU)

# GPU-specific parameters
gpu_batch_size: 360     # Process 360 months at a time on GPU
gpu_dtype: float64      # or float32 for 2x memory savings
```

## Implementation Strategy

### Phase 1: IPCA GPU Only (2-3 days)

**Goal**: Migrate only IPCA to GPU, keep everything else on CPU.

**Files to modify**:
1. Create `ipca_functions_gpu.py` (GPU version of IPCA)
2. Update `panel_processor.py` to call GPU version for IPCA
3. Keep Fama/DKKM on CPU with n_jobs=10

**Expected result**:
- Time: 93 seconds per iteration (19.6x speedup)
- CPU utilization: 10 cores for Fama/DKKM
- GPU utilization: 100% during IPCA (72 seconds)

### Phase 2: Add SDF GPU (2-3 days)

**Goal**: Also migrate SDF to GPU.

**Files to modify**:
1. Create `sdf_compute_gpu.py` (GPU version of SDF)
2. Update `panel_processor.py` to call GPU version for SDF
3. Keep Fama/DKKM on CPU with n_jobs=10

**Expected result**:
- Time: 84 seconds per iteration (21.8x speedup)
- CPU utilization: 10 cores for Fama/DKKM
- GPU utilization: 100% during IPCA+SDF (72.5 seconds)

## Code Example: Hybrid Implementation

```python
# panel_processor.py

class HybridPanelProcessor:
    """Processor that uses both CPU (n_jobs) and GPU."""

    def __init__(self, config):
        self.config = config

        # Check GPU availability
        self.use_gpu = config.use_gpu
        if self.use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                print(f"GPU detected: {cp.cuda.Device().name}")
                free, total = cp.cuda.Device().mem_info
                print(f"GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                print("Falling back to CPU-only mode")
                self.use_gpu = False

    def _compute_factors(self, panel, start, end, iter_num):
        """Compute factors using hybrid CPU+GPU."""

        factors_data = {}

        # === CPU-PARALLELIZED OPERATIONS ===
        print(f"Computing Fama-French on {self.config.n_jobs} CPU cores...")
        factors_data['ff_rets'] = fama.factors(
            fama.fama_french, panel,
            n_jobs=self.config.n_jobs, start=start, end=end
        )

        print(f"Computing Fama-MacBeth on {self.config.n_jobs} CPU cores...")
        factors_data['fm_rets'] = fama.factors(
            fama.fama_macbeth, panel,
            n_jobs=self.config.n_jobs, start=start, end=end
        )

        print(f"Computing DKKM on {self.config.n_jobs} CPU cores...")
        # DKKM computation with n_jobs...

        # === GPU OPERATIONS ===
        if self.use_gpu:
            print("Computing IPCA on GPU...")
            import ipca_functions_gpu as ipca_gpu

            ipca_lst = []
            for K in self.config.ipca_nfactors_lst:
                result = ipca_gpu.fit_ipca_360_gpu(
                    panel, K, self.config.N, start, end, 0, 0
                )
                ipca_lst.append(result)

            factors_data['ipca_lst'] = ipca_lst
        else:
            # Fallback: CPU parallelization over IPCA variants
            print(f"Computing IPCA on {self.config.n_jobs} CPU cores (no GPU)...")
            ipca_lst = Parallel(n_jobs=self.config.n_jobs)(
                delayed(ipca.fit_ipca_360)(panel, K, self.config.N, start, end, 0, 0)
                for K in self.config.ipca_nfactors_lst
            )
            factors_data['ipca_lst'] = ipca_lst

        return factors_data

    def _evaluate_months(self, panel, sdf_loop, factors_data, ...):
        """Evaluate months - optionally use GPU for SDF."""

        if self.use_gpu:
            # Use GPU-accelerated SDF
            import sdf_compute_gpu
            sdf_loop_gpu = sdf_compute_gpu.sdf_compute_gpu(N, T, arr_tuple)

            monthly_results = []
            for month in tqdm(months, desc="Evaluating months"):
                sdf_ret, max_sr, rp, cond_var = sdf_loop_gpu(month - 1, iter_num)
                # ... rest of evaluation ...
                monthly_results.append(result)
        else:
            # Use CPU version
            monthly_results = []
            for month in tqdm(months, desc="Evaluating months"):
                sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1, iter_num)
                # ... rest of evaluation ...
                monthly_results.append(result)

        return monthly_results
```

## Resource Utilization Timeline

### Sequential Hybrid (Recommended)

```
Component         | Time (s) | CPU Cores Used | GPU Used | Notes
------------------|----------|----------------|----------|------------------
Panel generation  |    6     |      1         |    No    | Serial operation
Fama-French       |    0.5   |     ~10        |    No    | Well-parallelized
Fama-MacBeth      |    0.5   |     ~10        |    No    | Well-parallelized
DKKM              |    0.25  |     ~10        |    No    | Well-parallelized
IPCA              |   72     |      0         |   Yes    | 100% GPU, CPU idle
Monthly eval      |    4     |      1         |    No    | Serial, could use GPU
------------------|----------|----------------|----------|------------------
TOTAL             |   83.25  |                |          | 21.8x speedup
```

**Observation**: CPU sits mostly idle during IPCA (72 sec). Could use this time for other tasks if you have multiple iterations to run.

### Concurrent Multi-Iteration Strategy

If you need to run multiple iterations, you can overlap them:

```python
# Run 2 iterations concurrently: one on CPU, one on GPU
import concurrent.futures

def run_iteration_cpu(iter_num):
    """Run iteration on CPU only (n_jobs=10)."""
    processor = PanelProcessor(config, use_gpu=False)
    return processor.process_iteration(iter_num, writer)

def run_iteration_gpu(iter_num):
    """Run iteration on GPU."""
    processor = PanelProcessor(config, use_gpu=True)
    return processor.process_iteration(iter_num, writer)

# Run iterations 0 and 1 concurrently
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    future_cpu = executor.submit(run_iteration_cpu, 0)
    future_gpu = executor.submit(run_iteration_gpu, 1)

    results = [future_cpu.result(), future_gpu.result()]
```

**Timeline**:
```
Time        CPU (iteration 0)           GPU (iteration 1)
0           Panel gen (6s)              Panel gen (6s)
6           Fama+DKKM (1.25s)           Fama+DKKM (1.25s)
7.25        IPCA CPU (1800s)            IPCA GPU (72s)
79.25       ...                         Monthly eval (4s)
83.25       ...                         DONE (iter 1)
1807        Monthly eval (4s)           Idle
1811        DONE (iter 0)               Idle
```

**Total for 2 iterations**: 1811 seconds (vs. 3642 seconds sequential)

**Speedup**: 2x (nearly perfect scaling)

## Troubleshooting

### Issue 1: GPU Out of Memory

**Symptom**:
```
cupy.cuda.memory.OutOfMemoryError
```

**Solutions**:
1. **Reduce batch size**: Process fewer months at once
2. **Use float32**: Half the memory usage
3. **Free memory between operations**:
   ```python
   mempool = cp.get_default_memory_pool()
   mempool.free_all_blocks()
   ```
4. **Fall back to CPU** for that operation

### Issue 2: CPU and GPU Competing for Memory

**Symptom**: System slowdown, swapping

**Solution**:
- Limit CPU parallelism when GPU is active:
  ```python
  if use_gpu:
      n_jobs_during_gpu = 2  # Reduce CPU load
  else:
      n_jobs_during_gpu = 10
  ```

### Issue 3: Thread Safety with joblib + GPU

**Symptom**: Crashes or incorrect results

**Solution**:
- Don't call GPU code from within joblib.Parallel
- Keep GPU operations separate from CPU parallel operations
- Use sequential hybrid (Option 1) instead of concurrent

## Performance Monitoring

### Track Resource Utilization

```python
import time
import psutil
import threading

class ResourceMonitor:
    """Monitor CPU and GPU usage during execution."""

    def __init__(self):
        self.monitoring = False
        self.stats = []

    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        import cupy as cp
        while self.monitoring:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)

            # GPU usage
            free, total = cp.cuda.Device().mem_info
            gpu_mem_used = (total - free) / 1e9

            self.stats.append({
                'time': time.time(),
                'cpu_avg': sum(cpu_percent) / len(cpu_percent),
                'cpu_cores': cpu_percent,
                'gpu_mem_gb': gpu_mem_used
            })

            time.sleep(0.5)

    def stop(self):
        self.monitoring = False
        self.thread.join()
        return self.stats

# Usage
monitor = ResourceMonitor()
monitor.start()

# Run your iteration
processor.process_iteration(0, writer)

stats = monitor.stop()

# Analyze
import pandas as pd
df = pd.DataFrame(stats)
print(f"Average CPU usage: {df['cpu_avg'].mean():.1f}%")
print(f"Peak GPU memory: {df['gpu_mem_gb'].max():.1f} GB")
```

## Cost-Benefit Analysis

### Investment

| Item | Cost | Notes |
|------|------|-------|
| GPU (RTX 4090) | $1,600 | One-time |
| Development (Phase 1-2) | $2,880 | 48 hours @ $60/hr |
| Electricity (3 years) | $200/yr | 350W GPU @ $0.12/kWh |
| **Total (3 years)** | **$5,080** | |

### Return

**Time savings per year**:
- 4 research projects
- 1600 iterations per project
- 6400 iterations total

**Baseline (n_jobs=10)**:
- 6400 iterations × 30 min = 3200 hours

**Hybrid (n_jobs=10 + GPU)**:
- 6400 iterations × 1.4 min = 149 hours

**Savings**: 3051 hours/year @ $60/hr = **$183,060/year**

**ROI**: $183k / $5k = **36:1**
**Payback**: 10 days

## Conclusion

**Recommended Setup**:
- Keep n_jobs=10 for CPU parallelization (Fama, DKKM)
- Add 1 GPU (RTX 4090, 24GB) for IPCA and SDF
- Use sequential hybrid architecture (simpler, 99% of maximum performance)

**Expected Results**:
- Development time: 4-6 days
- Speedup: 21.8x (30 min → 84 sec)
- Hardware cost: $1,600 (GPU only)
- ROI: 36:1

**Not Recommended**:
- Migrating Fama/DKKM to GPU (only 1.25x speedup, already efficient on CPU)
- Concurrent CPU+GPU (complex, minimal benefit)
- More than 1 GPU (unless running multiple iterations in parallel)
