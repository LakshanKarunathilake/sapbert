import torch

def check_cuda():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print("CUDA is available!")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        # Print details for each GPU
        for i in range(num_gpus):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
            print(f"  Device Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No CUDA GPUs are available.")

if __name__ == "__main__":
    check_cuda()
