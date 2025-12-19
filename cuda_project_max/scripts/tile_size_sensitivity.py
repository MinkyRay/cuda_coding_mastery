import os
import subprocess
import re
import matplotlib.pyplot as plt
import sys


SOURCE_FILE = "src/04_gemm_double_buffering.cu"  
OUTPUT_EXE = "gemm_test"
if sys.platform == "win32":
    OUTPUT_EXE += ".exe"


BLOCK_TILE_SIZES = [32, 64, 128]


M, N, K = 4096, 4096, 4096

NVCC_FLAGS = "-arch=sm_89 -O3" 
REGISTER_BLOCK_SIZE = 8 


def run_sensitivity_test():
    results = []
    
    print(f"Starting Block Tile Size Sensitivity Analysis (TM/TN fixed to {REGISTER_BLOCK_SIZE})...")
    print(f"Matrix Size: {M}x{N}x{K}\n")

    for size in BLOCK_TILE_SIZES:
       
        threads_per_dim = size // REGISTER_BLOCK_SIZE 
        threads_total = threads_per_dim * threads_per_dim # 例如 128/8 = 16. 16*16 = 256
        
        print(f"--- Testing Block: {size} x {size} (Total Threads: {threads_total}) ---")
        
       
        compile_cmd = f"nvcc {SOURCE_FILE} -o {OUTPUT_EXE} {NVCC_FLAGS} -D BM={size} -D BN={size} -D TM={REGISTER_BLOCK_SIZE} -D TN={REGISTER_BLOCK_SIZE}"
        
        ret = subprocess.call(compile_cmd, shell=True)
        if ret != 0:
            print(f"[Error] Compile failed for size {size}. Skipping...")
            results.append(0)
            continue
            
       
        run_cmd = f"{OUTPUT_EXE} {M} {N} {K} {threads_total}" 
        
       
        
        try:
            output = subprocess.check_output(f"{OUTPUT_EXE} {M} {N} {K} {threads_total}", shell=True).decode("utf-8", errors="ignore")
            match = re.search(r"Performance:\s+([\d\.]+)\s+GFLOPS", output)
            if match:
                gflops = float(match.group(1))
                results.append(gflops)
                print(f"-> Performance: {gflops:.2f} GFLOPS")
            else:
                print("-> Parse Error.")
                results.append(0)
        except Exception as e:
            print(f"-> Runtime Error: {e}. (Likely CUDA Launch Failure)")
            results.append(0)
            
    return results


def plot_sensitivity(results):
    print("\n========================================")
    print("Step 3: Plotting Results...")
    print("========================================")
    
    plt.figure(figsize=(10, 6))
    
    x_labels = [f"{s}x{s}" for s in BLOCK_TILE_SIZES]
    
    plt.plot(x_labels, results, marker='o', linewidth=2, color='darkred', label='RTX 4050 Performance')
    

    max_gflops = max(results)
    max_idx = results.index(max_gflops)
    plt.annotate(f'Peak: {max_gflops:.0f} GFLOPS', 
                 xy=(max_idx, max_gflops), 
                 xytext=(max_idx, max_gflops * 1.05),
                 fontsize=12,
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title(f"Impact of Register Block Size on GEMM Performance (N={N})", fontsize=14)
    plt.xlabel("Register Block Dimension (TM x TN)", fontsize=12)
    plt.ylabel("Performance (GFLOPS)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    output_img = "register_block_sensitivity.png"
    plt.savefig(output_img)
    print(f"Graph saved to {output_img}")



if __name__ == "__main__":
    
    data = run_sensitivity_test()
    plot_sensitivity(data)