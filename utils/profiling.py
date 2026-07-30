# TODO should be upgraded for better profiling but the idea is right

import time

import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.profiler import profile

def profile_data_loading(loader, device, num_batches=10):
    """
    Profiles data loading speed and memory usage.

    Args:
        loader: PyTorch DataLoader
        device: torch.device ("cuda" or "cpu")
        num_batches: number of batches to profile
    """
    print("Profiling Data Loading...")
    
    batch_times = []
    start_total = time.time()
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        start = time.time()
        
        # Move batch to device (simulate what happens in training)
        batch_on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(device, non_blocking=True)
            else:
                batch_on_device[k] = v  # keep non-tensor objects as is

        end = time.time()
        batch_times.append(end - start)
        
        # Print memory usage after moving to GPU
        if device == "cuda":
            print(
                f"Batch {i} - time: {end - start:.4f}s, "
                f"allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
                f"reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB"
            )
            torch.cuda.reset_peak_memory_stats()

    total_time = time.time() - start_total
    avg_time = sum(batch_times) / len(batch_times)
    print(f"\nData loading profiling finished for {num_batches} batches.")
    print(f"Average batch transfer time: {avg_time:.4f}s")
    print(f"Total time for {num_batches} batches: {total_time:.4f}s")

def profile_model(
        model, 
        loader, 
        device, 
        amp = True, 
        amp_dtype_str = "bfloat16",
        num_batches=5,
        num_warmup_batches=3, # Avoid compile, sync, copies HtoD, Lazy modules
        output_path = "profiling",
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        row_limit=30,
        export_chrome_trace=True,
        export_stacks=False,
        ):
    model.eval()
    device = torch.device(device)
    amp_dtype = getattr(torch, amp_dtype_str)
    os.makedirs(output_path, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    if num_warmup_batches > 0:
        print(f"Warmup on {num_warmup_batches} batch(es)...")
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= num_warmup_batches:
                    break
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)
                    else:
                        batch[k] = v
                with torch.autocast(device.type, enabled=amp, dtype=amp_dtype):
                    outputs = model(batch)
                del outputs
                torch.cuda.synchronize(device)
                print(f"Warmup batch {i}")

        # Reset memory stats after warmup so reported peak is only for profiled region
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as prof:
        for i, batch in enumerate(loader):
            if i >= num_batches:  # Only profile first batches
                print("Finished looping, printing the results:")
                break
            with torch.profiler.record_function("batch_to_device"):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(device, non_blocking=True)
                    else:
                        batch[k] = v
            with torch.profiler.record_function("model_forward"):
                with torch.autocast(device.type, enabled = amp, dtype= amp_dtype):
                    with torch.no_grad():
                        outputs = model(batch)
                    del outputs
            # torch.cuda.empty_cache()
            print(f"Batch {i}")
    
    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    # print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    # print(f"Peak CUDA memory reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB")
    # torch.cuda.reset_peak_memory_stats()
    # Build tables
    key_avg = prof.key_averages()

    tables = {
        "cuda_time_total": key_avg.table(sort_by="cuda_time_total", row_limit=row_limit)
        if device.type == "cuda"
        else None,
        "self_cuda_time_total": key_avg.table(sort_by="self_cuda_time_total", row_limit=row_limit)
        if device.type == "cuda"
        else None,
        "cpu_time_total": key_avg.table(sort_by="cpu_time_total", row_limit=row_limit),
        "self_cpu_time_total": key_avg.table(sort_by="self_cpu_time_total", row_limit=row_limit),
        "cuda_memory_usage": key_avg.table(sort_by="cuda_memory_usage", row_limit=row_limit)
        if device.type == "cuda" and profile_memory
        else None,
        "self_cuda_memory_usage": key_avg.table(sort_by="self_cuda_memory_usage", row_limit=row_limit)
        if device.type == "cuda" and profile_memory
        else None,
        "cpu_memory_usage": key_avg.table(sort_by="cpu_memory_usage", row_limit=row_limit)
        if profile_memory
        else None,
        "self_cpu_memory_usage": key_avg.table(sort_by="self_cpu_memory_usage", row_limit=row_limit)
        if profile_memory
        else None,
    }

    # Peak memory
    peak_cuda_allocated_mb = None
    peak_cuda_reserved_mb = None
    if device.type == "cuda":
        peak_cuda_allocated_mb = torch.cuda.max_memory_allocated(device) / 1e6
        peak_cuda_reserved_mb = torch.cuda.max_memory_reserved(device) / 1e6

    # Save tables
    txt_report_path = os.path.join(output_path, "profiler_report.txt")
    with open(txt_report_path, "w", encoding="utf-8") as f:
        f.write("=== PROFILER REPORT ===\n\n")
        f.write(f"device: {device}\n")
        f.write(f"amp: {amp}\n")
        f.write(f"amp_dtype: {amp_dtype_str}\n")
        f.write(f"num_batches: {num_batches}\n")
        f.write(f"record_shapes: {record_shapes}\n")
        f.write(f"profile_memory: {profile_memory}\n")
        f.write(f"with_stack: {with_stack}\n")
        f.write(f"with_flops: {with_flops}\n")
        f.write(f"row_limit: {row_limit}\n\n")

        if peak_cuda_allocated_mb is not None:
            f.write(f"Peak CUDA memory allocated: {peak_cuda_allocated_mb:.2f} MB\n")
            f.write(f"Peak CUDA memory reserved:  {peak_cuda_reserved_mb:.2f} MB\n\n")

        for name, table in tables.items():
            if table is None:
                continue
            f.write(f"=== SORT BY: {name} ===\n")
            f.write(table)
            f.write("\n\n")

    # Export chrome trace
    chrome_trace_path = None
    if export_chrome_trace:
        chrome_trace_path = os.path.join(output_path, "trace.json")
        prof.export_chrome_trace(chrome_trace_path)

    # Optional stack export
    stack_path = None
    if export_stacks and with_stack:
        stack_path = os.path.join(output_path, "stacks.txt")
        prof.export_stacks(stack_path, "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total")

    # Small machine-readable summary
    summary = {
        "device": str(device),
        "amp": amp,
        "amp_dtype": amp_dtype_str,
        "num_batches": num_batches,
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
        "with_flops": with_flops,
        "peak_cuda_memory_allocated_mb": peak_cuda_allocated_mb,
        "peak_cuda_memory_reserved_mb": peak_cuda_reserved_mb,
        "report_txt": txt_report_path,
        "chrome_trace": chrome_trace_path,
        "stack_file": stack_path,
    }

    summary_path = os.path.join(output_path, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== Profiling summary ===")
    print(f"Report: {txt_report_path}")
    if chrome_trace_path is not None:
        print(f"Chrome trace: {chrome_trace_path}")
    if stack_path is not None:
        print(f"Stacks: {stack_path}")
    if peak_cuda_allocated_mb is not None:
        print(f"Peak CUDA allocated: {peak_cuda_allocated_mb:.2f} MB")
        print(f"Peak CUDA reserved:  {peak_cuda_reserved_mb:.2f} MB")

    print("\n=== Top ops by CUDA time ===")
    if tables["cuda_time_total"] is not None:
        print(tables["cuda_time_total"])
    else:
        print("CUDA profiling not enabled.")

    print("\n=== Top ops by CUDA memory ===")
    if tables["cuda_memory_usage"] is not None:
        print(tables["cuda_memory_usage"])
    else:
        print("CUDA memory profiling not enabled.")

    return summary

def profile_model_training(model, loader, device, use_half_precision=True, ignore_index=-1, loss_fn=None):
    model.train()  # important: set to train mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)  # dummy optimizer
    scaler = GradScaler(enabled=use_half_precision)

    if loss_fn is None:
        loss_fn = lambda output, batch: torch.nn.functional.cross_entropy(
            output, batch["segment"], ignore_index=ignore_index
        )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        # record_shapes=True,
        profile_memory=True,
        # with_stack=True
        # on_trace_ready=
        
    ) as prof:
        for i, batch in enumerate(loader):
            if i >= 5:  # only profile first few batches
                break
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
                else:
                    batch[k] = v
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, enabled=use_half_precision):
                outputs = model(batch)
                loss = loss_fn(outputs, batch)
            scaler.scale(loss).backward()  # track gradient memory
            scaler.step(optimizer)  # optional for profiling
            scaler.update()
            print(f"Batch {i}: allocated {torch.cuda.memory_allocated() / 1e6:.2f} MB, "
                  f"reserved {torch.cuda.memory_reserved() / 1e6:.2f} MB")

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=15))
    print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    print(f"Peak CUDA memory reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB")
    torch.cuda.reset_peak_memory_stats()

def autobatch(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    dataloader: torch.utils.data.DataLoader,
    fraction: float = 0.60,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    default_batch_size: int = 2,
    ignore_index=-1,
    collate_fn=None,
    loss_fn=None,
) -> int:
    """Automatically find the largest safe batch size for point cloud training.

    Returns:
        Optimal batch size as an int.
    """
    prefix = "AutoBatch: "
    device = next(model.parameters()).device

    if device.type != "cuda":
        print(f"{prefix}CUDA not available, returning default batch size {default_batch_size}")
        return default_batch_size

    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    t = props.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    print(
        f"{prefix}{props.name}  total={t:.1f}G  reserved={r:.2f}G  "
        f"allocated={a:.2f}G  free={f:.2f}G"
    )

    # batch_sizes = [1, 2, 4, 8, 16] if t < 24 else [1, 2, 4, 8, 16, 32]
    batch_sizes = [1, 2, 4, 6, 8, 10, 12]
    scaler_enabled = amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device=device.type, enabled=scaler_enabled)

    if loss_fn is None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss_fn = lambda output, batch: criterion(output, batch["segment"])

    results = []   # (batch_size, peak_mem_GiB)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        batch = None
        try:
            loader = dataloader(dataset, bs, collate_fn=collate_fn, shuffle=True)
            batch = next(iter(loader))
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
                else:
                    batch[k] = v
                    
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(batch)
                loss = loss_fn(output, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            peak = torch.cuda.max_memory_allocated(device) / gb
            results.append((bs, peak))
            print(f"{prefix}  batch={bs:3d}  peak={peak:.2f}G / {t:.1f}G")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"{prefix}  batch={bs:3d}  OOM — stopping probe")
            else:
                print(f"{prefix}  batch={bs:3d}  error: {e} — stopping probe")
            torch.cuda.empty_cache()
            break
        finally:
            if batch is not None:
                del batch
            torch.cuda.empty_cache()

    if not results:
        print(f"{prefix}All batch sizes OOM, returning default {default_batch_size}")
        return default_batch_size

    xs = np.array([x for x, _ in results], dtype=float)
    ys = np.array([y for _, y in results], dtype=float)

    if len(xs) >= 2:
        p = np.polyfit(xs, ys, deg=1)              # memory = p[0]*batch + p[1]
        target = f * fraction + a + r              # absolute GiB budget
        b = int((target - p[1]) / p[0])            # solve for batch size
        # clamp to last safe probed size if extrapolation overshoots
        last_safe = int(results[-1][0])
        if b < 1:
            b = 1
        elif b > last_safe * 4:                    # don't extrapolate too far
            b = last_safe
        predicted_frac = np.polyval(p, b) / t
        print(
            f"{prefix}Optimal batch size: {b}  "
            f"(predicted {np.polyval(p, b):.2f}G / {t:.1f}G = {predicted_frac*100:.0f}%)"
        )
    else:
        b = int(results[-1][0])
        print(f"{prefix}Only one data point, using batch size {b}")

    return b

# def autobatch(
#     model: nn.Module,
#     max_points: int = 16384,
#     in_channels: int = 6,
#     num_classes: int = 12,
#     fraction: float = 0.60,
#     amp: bool = True,
#     amp_dtype: torch.dtype = torch.float16,
#     default_batch_size: int = 2,
#     loss_fn=None,
#     make_batch_fn=None,
# ) -> int:
#     """Automatically find the largest safe batch size for point cloud training.

#     Runs a forward+backward pass with synthetic data at increasing batch sizes,
#     fits a linear model to memory usage, and returns the batch size that stays
#     within `fraction` of free CUDA memory.

#     Args:
#         model: The model to profile (moved to CUDA before calling).
#         max_points: Points per sample (should match SmartPointSampler.max_points).
#         in_channels: Feature dimension (coord+color = 6 by default).
#         num_classes: Number of output classes (used by default loss_fn only).
#         fraction: Target fraction of *free* CUDA memory to use (default 0.60).
#         amp: Whether to run with AMP (matches training setting).
#         amp_dtype: AMP dtype (float16 or bfloat16).
#         default_batch_size: Fallback if profiling fails.
#         loss_fn: Optional callable (output, batch) -> scalar Tensor. If None,
#             assumes model returns [N, num_classes] logits and uses CrossEntropyLoss.
#         make_batch_fn: Optional callable (batch_size) -> dict. If None, generates
#             a standard PTv3 batch with coord/feat/grid_coord/segment/offset.

#     Returns:
#         Optimal batch size as an int.
#     """
#     prefix = "AutoBatch: "
#     device = next(model.parameters()).device

#     if device.type != "cuda":
#         print(f"{prefix}CUDA not available, returning default batch size {default_batch_size}")
#         return default_batch_size

#     gb = 1 << 30
#     props = torch.cuda.get_device_properties(device)
#     t = props.total_memory / gb
#     r = torch.cuda.memory_reserved(device) / gb
#     a = torch.cuda.memory_allocated(device) / gb
#     f = t - (r + a)
#     print(
#         f"{prefix}{props.name}  total={t:.1f}G  reserved={r:.2f}G  "
#         f"allocated={a:.2f}G  free={f:.2f}G"
#     )

#     # batch_sizes = [1, 2, 4, 8, 16] if t < 24 else [1, 2, 4, 8, 16, 32]
#     batch_sizes = [1, 2, 3, 4, 5]
#     scaler_enabled = amp and amp_dtype == torch.float16
#     scaler = torch.amp.GradScaler(device=device.type, enabled=scaler_enabled)

#     if loss_fn is None:
#         criterion = nn.CrossEntropyLoss()
#         loss_fn = lambda output, batch: criterion(output, batch["segment"])

#     def _default_make_batch(bs):
#         n = max_points * bs
#         coord = torch.rand(n, 3, device=device) * 10.0
#         feat = torch.rand(n, in_channels, device=device)
#         grid_coord = (coord / 0.05).floor().int()
#         segment = torch.randint(0, num_classes, (n,), device=device)
#         offset = torch.arange(max_points, n + 1, max_points, device=device).int()
#         return {"coord": coord, "feat": feat, "grid_coord": grid_coord,
#                 "segment": segment, "offset": offset}

#     _make_batch = make_batch_fn if make_batch_fn is not None else _default_make_batch

#     results = []   # (batch_size, peak_mem_GiB)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     model.train()

#     for bs in batch_sizes:
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats(device)
#         batch = None
#         try:
#             batch = _make_batch(bs)
#             with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
#                 output = model(batch)
#                 loss = loss_fn(output, batch)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)

#             peak = torch.cuda.max_memory_allocated(device) / gb
#             results.append((bs, peak))
#             print(f"{prefix}  batch={bs:3d}  peak={peak:.2f}G / {t:.1f}G")

#         except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
#             if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
#                 print(f"{prefix}  batch={bs:3d}  OOM — stopping probe")
#             else:
#                 print(f"{prefix}  batch={bs:3d}  error: {e} — stopping probe")
#             torch.cuda.empty_cache()
#             break
#         finally:
#             if batch is not None:
#                 del batch
#             torch.cuda.empty_cache()

#     if not results:
#         print(f"{prefix}All batch sizes OOM, returning default {default_batch_size}")
#         return default_batch_size

#     xs = np.array([x for x, _ in results], dtype=float)
#     ys = np.array([y for _, y in results], dtype=float)

#     if len(xs) >= 2:
#         p = np.polyfit(xs, ys, deg=1)              # memory = p[0]*batch + p[1]
#         target = f * fraction + a + r              # absolute GiB budget
#         b = int((target - p[1]) / p[0])            # solve for batch size
#         # clamp to last safe probed size if extrapolation overshoots
#         last_safe = int(results[-1][0])
#         if b < 1:
#             b = 1
#         elif b > last_safe * 4:                    # don't extrapolate too far
#             b = last_safe
#         predicted_frac = np.polyval(p, b) / t
#         print(
#             f"{prefix}Optimal batch size: {b}  "
#             f"(predicted {np.polyval(p, b):.2f}G / {t:.1f}G = {predicted_frac*100:.0f}%)"
#         )
#     else:
#         b = int(results[-1][0])
#         print(f"{prefix}Only one data point, using batch size {b}")

#     return b


def batch_memory(batch: dict) -> float:
    total_bytes = 0

    for val in batch.values():
        if isinstance(val, torch.Tensor):
            total_bytes += val.numel() * val.element_size()

    mb_memory = total_bytes / (1024 ** 2)
    print(f"Size of a batch: {mb_memory:.2f} MB")

    return mb_memory


def get_model_size(model: nn.Module) -> tuple[int, float]:
    """
    Computes the total number of parameters and the estimated model size in MB.
    
    Assumes that each parameter is stored as a 32-bit float (4 bytes).
    
    Args:
        model (nn.Module): The PyTorch model.
    
    Returns:
        total_params (int): Total number of parameters in the model.
        size_MB (float): Estimated size of the model in megabytes.
    """
    # Sum up all the parameters (both trainable and non-trainable)
    total_params = sum(p.numel() for p in model.parameters())
    size_in_bytes = total_params * 4 # Assuming 4 bytes per parameter (float32)
    
    # To megabytes.
    size_MB = size_in_bytes / (1024 ** 2)

    print(f"Model Size: {size_MB:.2f} MB")
    print(f"Total Parameters: {total_params:,}")
    
    return total_params, size_MB